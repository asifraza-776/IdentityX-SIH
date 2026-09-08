from datetime import datetime

REQUIRED_FIELDS = {
    "passport": ["name", "passport_number", "dob", "expiry"],
    "visa": ["name", "visa_number", "visa_type", "entry_validation", "stay_duration"],
    "national_id": ["name", "id_number", "dob"],
    "driving_license": ["name", "license_number", "dob", "expiry"],
    "permit": ["name", "permit_number", "permit_type", "expiry"],
    "aadhaar": ["name", "aadhaar_number", "dob"],
    "pan_card": ["name", "pan_number", "dob"],
    "college_id": ["name", "roll_number"],
    "marksheet": ["name"],
    "voter_id": ["name", "voter_id_number"],
    "other": [],
}


def validate_document(
    extracted_fields: dict,
    mrz_consistency: dict,
    document_type: str = "passport"
) -> dict:
    """
    Applies business rules to validate the document.
    """
    required = REQUIRED_FIELDS.get(document_type, REQUIRED_FIELDS.get("other", []))
    expiry_required = "expiry" in required

    checks = {
        "required_fields_present": False,
        "expiry_valid": True if not expiry_required else False,
        "mrz_consistent": (
            mrz_consistency.get("consistent", False)
            if mrz_consistency.get("status") != "NOT_APPLICABLE" else True
        )
    }
    
    warnings = []
    
    # 1. Check required fields
    missing = [field for field in required if not extracted_fields.get(field)]
    if not missing:
        checks["required_fields_present"] = True
    else:
        warnings.append(f"Missing fields: {', '.join(missing)}")
        
    # 2. Check Expiry Date if applicable
    expiry_str = extracted_fields.get("expiry")
    if expiry_str and expiry_required:
        try:
            # Assuming DD-MM-YYYY format from our OCR/MRZ parser
            expiry_date = datetime.strptime(expiry_str, "%d-%m-%Y")
            if expiry_date > datetime.now():
                checks["expiry_valid"] = True
            else:
                warnings.append("Document has expired.")
        except ValueError:
            warnings.append("Invalid expiry date format.")
            
    # Combine results
    status = "VALID"
    if (expiry_required and not checks["expiry_valid"]) or not checks["mrz_consistent"]:
        status = "SUSPICIOUS"
    elif not checks["required_fields_present"]:
        status = "INCOMPLETE" if required else "VALID"

    if status == "INCOMPLETE" and not required:
        status = "VALID"

    return {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "mrz_mismatches": mrz_consistency.get("mismatches", [])
    }
