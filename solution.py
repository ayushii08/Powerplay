"""
Converts unstructured text to structured JSON for Powerplay.

Design Philosophy: Defensive extraction with explicit uncertainty handling.
LLMs trained to be helpful; construction needs systems that refuse to guess.

Regional Language Support: Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# SCHEMA DEFINITION


class ProcurementOrder(BaseModel):
    """Strict schema for construction material orders."""
    material_name: str = Field(..., description="Material being ordered (e.g., 'cement bags', 'TMT steel bars')")
    quantity: Optional[float] = Field(None, description="Quantity of material")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    project_name: Optional[str] = Field(None, description="Project or site name")
    location: Optional[str] = Field(None, description="Delivery location")
    urgency: str = Field(..., description="Urgency level: low, medium, high")
    deadline: Optional[str] = Field(None, description="ISO format date (YYYY-MM-DD)")
    
    # Metadata for debugging and improvement
    needs_review: bool = Field(False, description="Flag for human review")
    extraction_notes: Optional[str] = Field(None, description="Additional context")
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator('quantity')
    def validate_quantity(cls, v):
        """Quantity must be positive and realistic."""
        if v is not None and (v <= 0 or v > 100000):
            raise ValueError(f"Quantity {v} outside realistic bounds (0-100,000)")
        return v
    
    @field_validator('urgency')
    def validate_urgency(cls, v):
        """Urgency must be one of three levels."""
        if v not in ['low', 'medium', 'high']:
            raise ValueError(f"Urgency must be 'low', 'medium', or 'high', got '{v}'")
        return v
    
    @field_validator('deadline')
    def validate_deadline(cls, v):
        """Deadline must be future date in ISO format."""
        if v is not None:
            try:
                deadline_date = datetime.fromisoformat(v)
                if deadline_date.date() < datetime.now().date():
                    raise ValueError(f"Deadline {v} is in the past")
            except ValueError as e:
                raise ValueError(f"Invalid deadline format: {e}")
        return v


# DOMAIN PREPROCESSING - WITH REGIONAL LANGUAGE SUPPORT


class DomainPreprocessor:
    """Augments raw input with construction domain knowledge and regional language hints."""
    
    # Construction-specific terminology mappings
    MATERIAL_HINTS = {
        r'\b(\d+)mm\b': r'\1mm (diameter specification for TMT/rebar steel)',
        r'\bOPC\s*\d+\b': r'\g<0> (Ordinary Portland Cement grade)',
        r'\bTMT\b': 'TMT (Thermo-Mechanically Treated steel bars)',
        r'\bM-sand\b': 'M-sand (Manufactured sand for construction)',
        r'\btor\s+steel\b': 'tor steel (Twisted steel bars for reinforcement)',
        r'\bRMC\b': 'RMC (Ready-Mix Concrete)',
    }
    
    # Common Indian construction brands
    BRANDS = ['Ultratech', 'ACC', 'Ambuja', 'JSW', 'Tata', 'Birla']
    
    # Multi-regional temporal phrases covering 9 Indian languages
    TEMPORAL_HINTS = {
        # Hindi (North India)
        r'\bkal\s+tak\b': 'kal tak (Hindi: by tomorrow)',
        r'\bkal\s+subah\b': 'kal subah (Hindi: tomorrow morning)',
        r'\bkal\b': 'kal (Hindi: tomorrow)',
        r'\bparso\b': 'parso (Hindi: day after tomorrow)',
        r'\baaj\b': 'aaj (Hindi: today)',
        r'\babhi\b': 'abhi (Hindi: now/immediately)',
        
        # Tamil (Tamil Nadu)
        r'\bnalaiki\b': 'nalaiki (Tamil: tomorrow)',
        r'\binniki\b': 'inniki (Tamil: today)',
        r'\bippo\b': 'ippo (Tamil: now)',
        r'\brathri\b': 'rathri (Tamil: night)',
        r'\bvennum\b': 'vennum (Tamil: needed)',
        r'\bpannunga\b': 'pannunga (Tamil: please send)',
        
        # Telugu (Andhra Pradesh, Telangana)
        r'\brepu\b': 'repu (Telugu: tomorrow)',
        r'\bindu\b': 'indu (Telugu: today)',
        r'\bkavali\b': 'kavali (Telugu: needed)',
        r'\binkeppudu\b': 'inkeppudu (Telugu: now/immediately)',
        r'\bpampinchandi\b': 'pampinchandi (Telugu: please send)',
        
        # Kannada (Karnataka)
        r'\bnaale\b': 'naale (Kannada: tomorrow)',
        r'\bivaga\b': 'ivaga (Kannada: now)',
        r'\bbeku\b': 'beku (Kannada: needed)',
        r'\balli\b': 'alli (Kannada: there/at that place)',
        
        # Malayalam (Kerala)
        r'\bnale\b': 'nale (Malayalam: tomorrow)',
        r'\binnu\b': 'innu (Malayalam: today)',
        r'\bippol\s+thanne\b': 'ippol thanne (Malayalam: right now)',
        r'\bippol\b': 'ippol (Malayalam: now)',
        r'\bvenam\b': 'venam (Malayalam: needed)',
        r'\bvegam\b': 'vegam (Malayalam: urgently/quickly)',
        r'\bkodukkanam\b': 'kodukkanam (Malayalam: must give/send)',
        
        # Bengali (West Bengal)
        r'\baajke\b': 'aajke (Bengali: today)',
        r'\bekhoni\b': 'ekhoni (Bengali: now)',
        r'\bporjonto\b': 'porjonto (Bengali: by/until)',
        r'\bdorkar\b': 'dorkar (Bengali: needed)',
        r'\bpathao\b': 'pathao (Bengali: send)',
        
        # Gujarati (Gujarat)
        r'\baaje\b': 'aaje (Gujarati: today)',
        r'\baavkaal\b': 'aavkaal (Gujarati: tomorrow)',
        r'\bturant\b': 'turant (Gujarati: immediately)',
        r'\bjoiye\b': 'joiye (Gujarati: needed)',
        r'\bmoklo\b': 'moklo (Gujarati: send)',
        r'\bsaanje\b': 'saanje (Gujarati: evening)',
        
        # Marathi (Maharashtra)
        r'\budya\b': 'udya (Marathi: tomorrow)',
        r'\bpahije\b': 'pahije (Marathi: needed)',
        r'\blaagech\b': 'laagech (Marathi: immediately)',
        r'\bdupari\b': 'dupari (Marathi: afternoon)',
        r'\bvajta\b': 'vajta (Marathi: o\'clock)',
        
        # Punjabi (Punjab)
        r'\bajj\b': 'ajj (Punjabi: today)',
        r'\bhun\b': 'hun (Punjabi: now)',
        r'\bagley\b': 'agley (Punjabi: next)',
        r'\bshaam\b': 'shaam (Punjabi: evening)',
        r'\bbaje\b': 'baje (Punjabi: o\'clock)',
    }
    
    # Common honorifics across regions (to be recognized but not extracted)
    HONORIFICS = {
        r'\bbhaiya\b': 'bhaiya (Hindi: brother/boss)',
        r'\banna\b': 'anna (Tamil: elder brother)',
        r'\bchettan\b': 'chettan (Malayalam: elder brother)',
        r'\bdada\b': 'dada (Bengali: elder brother)',
        r'\bbhai\b': 'bhai (Gujarati/Hindi: brother)',
        r'\byaar\b': 'yaar (Punjabi/Hindi: friend)',
    }
    
    def preprocess(self, text: str) -> str:
        """Add domain context hints to raw input."""
        enhanced = text
        
        # Add material specification hints
        for pattern, replacement in self.MATERIAL_HINTS.items():
            enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)
        
        # Add temporal phrase translations
        for pattern, replacement in self.TEMPORAL_HINTS.items():
            if re.search(pattern, enhanced, re.IGNORECASE):
                enhanced = f"{enhanced} [{replacement}]"
        
        # Add honorific context (helps LLM understand these are informal markers)
        for pattern, replacement in self.HONORIFICS.items():
            if re.search(pattern, enhanced, re.IGNORECASE):
                enhanced = f"{enhanced} [Note: '{pattern.strip(r'\b')}' is an honorific]"
                break  # Only add note once
        
        return enhanced


# URGENCY CLASSIFICATION (RULE-BASED) - WITH REGIONAL KEYWORDS


class UrgencyClassifier:
    """Explicit rule-based urgency classification with regional language support."""
    
    HIGH_KEYWORDS = [
        # English
        'urgent', 'asap', 'immediately', 'emergency', 'stopped', 'critical', 'urgnt',
        # Hindi
        'turant', 'abhi', 'foran', 'urgenthai',
        # Tamil
        'ippo', 'urgenta',
        # Telugu
        'inkeppudu', 'urgentga',
        # Kannada
        'urgentagi', 'ivaga',
        # Malayalam
        'vegam', 'ippol thanne',
        # Bengali
        'ekhoni', 'turoturo',
        # Gujarati
        'turant', 'atyare',
        # Marathi
        'laagech', 'tatkal',
        # Punjabi
        'hun', 'furan',
        # Common phrases
        'work stopped', 'kaj bondho', 'site alli work stop', 'kam thayu', 'kamai band'
    ]
    
    MEDIUM_KEYWORDS = [
        # English
        'soon', 'quick', 'quickly', 'fast', 'priority', 'needed',
        # Cross-regional medium urgency
        'jaldi', 'shighra', 'bega', 'taratari', 'joldi'
    ]
    
    @classmethod
    def classify(cls, text: str, deadline: Optional[str]) -> str:
        """
        Urgency rules (explicit and debuggable):
        - HIGH: Contains urgent keywords OR deadline < 3 days
        - MEDIUM: Contains medium keywords OR deadline 3-7 days OR ambiguous
        - LOW: Deadline > 7 days OR no urgency indicators
        """
        text_lower = text.lower()
        
        # Check for high urgency keywords
        if any(kw in text_lower for kw in cls.HIGH_KEYWORDS):
            return 'high'
        
        # Check deadline-based urgency
        if deadline:
            try:
                deadline_date = datetime.fromisoformat(deadline)
                days_until = (deadline_date - datetime.now()).days
                
                if days_until < 3:
                    return 'high'
                elif days_until <= 7:
                    return 'medium'
                else:
                    return 'low'
            except ValueError:
                pass  # Invalid deadline, fall through
        
        # Check for medium urgency keywords
        if any(kw in text_lower for kw in cls.MEDIUM_KEYWORDS):
            return 'medium'
        
        # Default: medium (conservative middle ground for ambiguous cases)
        return 'medium'


# OUTPUT VALIDATION


class OutputValidator:
    """Post-LLM validation with domain-specific sanity checks."""
    
    # Valid unit-material combinations
    UNIT_MATERIAL_RULES = {
        'bags': ['cement', 'sand', 'aggregate'],
        'units': ['steel', 'bar', 'rod', 'brick', 'block'],
        'truckloads': ['sand', 'aggregate', 'soil', 'gravel'],
        'kg': ['steel', 'cement', 'rod', 'bar'],
        'tons': ['steel', 'cement', 'sand', 'aggregate'],
    }
    
    @classmethod
    def validate_domain_logic(cls, data: Dict[str, Any], original_text: str) -> tuple[bool, List[str]]:
        """
        Semantic validation beyond schema checking.
        Returns: (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check unit-material coherence - handle None values
        unit = data.get('unit')
        material = data.get('material_name')
        
        if unit and material:
            unit_lower = unit.lower() if isinstance(unit, str) else ''
            material_lower = material.lower() if isinstance(material, str) else ''
            
            if unit_lower in cls.UNIT_MATERIAL_RULES:
                valid_materials = cls.UNIT_MATERIAL_RULES[unit_lower]
                if not any(mat in material_lower for mat in valid_materials):
                    warnings.append(f"Unit '{unit}' typically not used with '{material}'")
        
        # Hallucination detection: check if project_name exists but no proper nouns in input
        project = data.get('project_name')
        if project and project.lower() not in ['none', 'null']:
            # Check if project name or similar capitalized words in original text
            if not re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', original_text):
                warnings.append(f"Project name '{project}' not found in input text - possible hallucination")
        
        # Check if deadline was extracted but no temporal words in input
        deadline = data.get('deadline')
        temporal_patterns = [r'\bby\b', r'\buntil\b', r'\bbefore\b', r'\bdeadline\b', 
                            r'\d+\s*(day|week|month)', r'\bkal\b', r'\bnalaiki\b', 
                            r'\brepu\b', r'\bnaale\b', r'urgent']
        if deadline:
            has_temporal = any(re.search(p, original_text, re.IGNORECASE) for p in temporal_patterns)
            if not has_temporal:
                warnings.append("Deadline extracted but no temporal reference in input")
        
        return len(warnings) == 0, warnings


# MAIN EXTRACTION ENGINE


class ProcurementExtractor:
    """Orchestrates LLM-based extraction with retry logic and validation."""
    
    def __init__(self):
        self.preprocessor = DomainPreprocessor()
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Defensive system prompt with explicit constraints and regional language support."""
        return """You are a construction procurement data extractor for Powerplay, India's #1 construction app.

Your task is to extract procurement data and return it as a JSON object.

CRITICAL RULES:
1. If a field is NOT explicitly stated in the input, return null - NEVER infer or guess
2. Output ONLY these fields: material_name, quantity, unit, project_name, location, deadline
3. Do NOT add fields like supplier_name, cost, estimated_price, material_grade
4. If multiple materials mentioned, extract ONLY the first material
5. For deadlines: convert to ISO format (YYYY-MM-DD); if ambiguous or relative ("month end", "soon", "tomorrow"), return null
6. Unit must be one of: bags, units, truckloads, kg, tons

REGIONAL LANGUAGE SUPPORT:
- You will receive inputs in mixed English + regional Indian languages (Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi)
- Common temporal terms: "kal/nalaiki/repu/naale/nale" (tomorrow), "aaj/inniki/indu" (today), "abhi/ippo/ekhoni" (now)
- Common urgency terms: "urgent/vegam/turant/laagech" (urgent)
- Honorifics (ignore these): "bhaiya/anna/chettan/dada/boss/bhai/yaar" (brother/boss/friend)
- Extract meaning and material_name in English, but recognize regional terms

EXAMPLES (JSON format):

Input: "50 Ultratech cement bags for Phoenix Project by March 15"
Output: {"material_name": "Ultratech cement bags", "quantity": 50, "unit": "bags", "project_name": "Phoenix Project", "location": null, "deadline": "2025-03-15"}

Input: "need cement urgent"
Output: {"material_name": "cement", "quantity": null, "unit": null, "project_name": null, "location": null, "deadline": null}

Input: "bhaiya kal subah 25 bag Ultratech bhej do urgent"
Output: {"material_name": "Ultratech cement bags", "quantity": 25, "unit": "bags", "project_name": null, "location": null, "deadline": null}

Remember: Null is honesty. Guessing causes ₹50,000 deliveries to wrong sites. Always return valid JSON."""
    
    def extract(self, text: str, retry_on_failure: bool = True) -> Dict[str, Any]:
        """
        Main extraction with retry logic.
        Returns: Dict with extraction results + metadata
        """
        # Step 1: Domain preprocessing
        enhanced_text = self.preprocessor.preprocess(text)
        
        # Step 2: LLM extraction (with retry)
        raw_output = self._call_llm(enhanced_text)
        
        if raw_output is None and retry_on_failure:
            # Retry once with error feedback
            raw_output = self._call_llm(
                enhanced_text, 
                error_context="Previous attempt failed. Ensure strict JSON format with only specified fields."
            )
        
        # Step 3: Validate and classify urgency
        if raw_output:
            try:
                # Override urgency with rule-based classification
                raw_output['urgency'] = UrgencyClassifier.classify(text, raw_output.get('deadline'))
                
                # Validate schema
                validated = ProcurementOrder(**raw_output)
                
                # Domain sanity checks
                is_valid, warnings = OutputValidator.validate_domain_logic(raw_output, text)
                
                if warnings:
                    validated.needs_review = True
                    validated.extraction_notes = "; ".join(warnings)
                
                return validated.model_dump()
                
            except ValidationError as e:
                # Schema validation failed
                return self._create_fallback_output(text, f"Validation error: {str(e)}")
        
        # Complete failure after retry
        return self._create_fallback_output(text, "LLM extraction failed")
    
    def _call_llm(self, text: str, error_context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Call OpenAI with structured output."""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if error_context:
                messages.append({"role": "user", "content": f"Error from previous attempt: {error_context}"})
            
            messages.append({"role": "user", "content": f"Extract procurement data as JSON from: {text}"})
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective for production
                messages=messages,
                temperature=0,  # Deterministic outputs
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None
    
    def _create_fallback_output(self, text: str, error_msg: str) -> Dict[str, Any]:
        """Graceful degradation: return partial extraction with review flag."""
        return {
            "material_name": self._extract_material_fallback(text),
            "quantity": None,
            "unit": None,
            "project_name": None,
            "location": None,
            "urgency": UrgencyClassifier.classify(text, None),
            "deadline": None,
            "needs_review": True,
            "extraction_notes": error_msg
        }
    
    def _extract_material_fallback(self, text: str) -> str:
        """Regex-based material extraction as fallback."""
        common_materials = ['cement', 'steel', 'sand', 'aggregate', 'brick', 'rod', 'bar', 'TMT']
        text_lower = text.lower()
        
        for material in common_materials:
            if material.lower() in text_lower:
                return material
        
        return "unknown material"


# MAIN EXECUTION


def main():
    """Process test inputs and generate outputs."""
    
    # Read test inputs
    try:
        with open('test_inputs.txt', 'r', encoding='utf-8') as f:
            test_inputs = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: test_inputs.txt not found")
        return
    
    # Initialize extractor
    extractor = ProcurementExtractor()
    
    # Process each input
    results = []
    print(f"\n{'='*70}")
    print(f"POWERPLAY PROCUREMENT EXTRACTOR - Multi-Regional Language Edition")
    print(f"{'='*70}\n")
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"[{i}/{len(test_inputs)}] Processing: {input_text[:60]}...")
        
        result = extractor.extract(input_text)
        result['input_text'] = input_text  # Include original for reference
        results.append(result)
        
        if result['needs_review']:
            print(f"  ⚠️  Flagged for review: {result['extraction_notes']}")
        else:
            print(f"  ✓ Extracted: {result['material_name']}, qty={result['quantity']}, urgency={result['urgency']}")
    
    # Save outputs
    with open('outputs.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Summary statistics
    total = len(results)
    flagged = sum(1 for r in results if r['needs_review'])
    complete = sum(1 for r in results if r['quantity'] is not None and r['material_name'] != 'unknown material')
    high_urgency = sum(1 for r in results if r['urgency'] == 'high')
    
    print(f"\n{'='*70}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total inputs: {total}")
    print(f"Complete extractions: {complete} ({complete/total*100:.1f}%)")
    print(f"Flagged for review: {flagged} ({flagged/total*100:.1f}%)")
    print(f"High urgency cases: {high_urgency}")
    print(f"Outputs saved to: outputs.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()