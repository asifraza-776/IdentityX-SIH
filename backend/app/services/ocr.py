import cv2
import numpy as np
import easyocr
import re

from app.services.mrz import parse_mrz, verify_mrz_consistency

reader = None


DOCUMENT_FIELDS = {
    "passport": ["name", "passport_number", "nationality", "dob", "expiry", "sex"],
    "visa": ["name", "visa_number", "visa_type", "entry_validation", "stay_duration"],
    "national_id": ["name", "id_number", "nationality", "dob", "expiry", "sex"],
    "driving_license": ["name", "license_number", "dob", "expiry", "issue_date", "class"],
    "permit": ["name", "permit_number", "permit_type", "issue_date", "expiry"],
    "aadhaar": ["name", "aadhaar_number", "dob", "gender"],
    "pan_card": ["name", "pan_number", "father_name", "dob"],
    "college_id": ["name", "roll_number", "course", "dob"],
    "marksheet": ["name", "roll_number", "passing_year", "percentage"],
    "voter_id": ["name", "voter_id_number", "dob", "father_name"],
    "other": ["name", "dob"],
}


def process_document_ocr(image_bytes: bytes, document_type: str = "passport") -> dict:
    global reader

    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not decode image."}

    # OCR
    results = reader.readtext(img, detail=0)

    document_type = document_type if document_type in DOCUMENT_FIELDS else "passport"
    parsed_data = parse_document_text(results, document_type)

    # MRZ lines
    mrz_lines = [
        line.strip().upper()
        for line in results
        if '<' in line and len(line.replace(" ", "")) > 20
    ]

    mrz_data = parse_mrz(mrz_lines, document_type)

    # Prefer MRZ values when available
    if mrz_data.get("status") == "success":
        for field in [
            "passport_number",
            "visa_number",
            "id_number",
            "license_number",
            "permit_number",
            "dob",
            "expiry",
            "nationality",
            "sex",
            "surname",
            "given_names",
            "name"
        ]:
            if field in parsed_data and mrz_data.get(field):
                parsed_data[field] = mrz_data[field]

    # Compare normal OCR with MRZ
    verification = verify_mrz_consistency(
        parsed_data,
        mrz_data,
        document_type
    )

    return {
        "raw_text_segments": results,
        "extracted_fields": parsed_data,
        "mrz_lines": mrz_lines,
        "mrz": mrz_data,
        "verification": verification,
        "status": "success" if results else "failed"
    }


def parse_document_text(text_lines: list, document_type: str) -> dict:
    """Extract fields using the selected document's labels and number patterns."""
    data = {field: None for field in DOCUMENT_FIELDS[document_type]}
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text_lines if line.strip()]
    full_text = " ".join(lines).upper()

    def value_after_label(labels):
        for index, line in enumerate(lines):
            upper_line = line.upper()
            for label in labels:
                match = re.search(rf"\b{re.escape(label)}\b\s*[:#-]?\s*(.*)$", upper_line)
                if match and match.group(1).strip():
                    return match.group(1).strip()
                if re.fullmatch(rf".*\b{re.escape(label)}\b.*", upper_line) and index + 1 < len(lines):
                    return lines[index + 1].upper()
        return None

    field_labels = {
        "name": [
            "FULL NAME", "NAME", "SURNAME", "APPLICANT NAME", "HOLDER NAME",
            "CANDIDATE NAME", "STUDENT NAME", "NAME OF CANDIDATE", "MR./MS.",
            "MR.", "MS.", "MR/MS", "SHRI", "SMT.", "THIS IS TO CERTIFY THAT"
        ],
        "passport_number": ["PASSPORT NO", "PASSPORT NUMBER"],
        "visa_number": ["VISA NO", "VISA NUMBER"],
        "visa_type": ["VISA TYPE", "TYPE OF VISA", "CATEGORY"],
        "entry_validation": ["ENTRY VALIDITY", "VALID UNTIL", "NUMBER OF ENTRIES", "ENTRIES"],
        "stay_duration": ["DURATION OF STAY", "STAY DURATION", "DURATION", "DAYS"],
        "id_number": ["AADHAAR NO", "AADHAAR NUMBER", "ID NO", "ID NUMBER", "IDENTITY NO"],
        "license_number": ["LICENSE NO", "LICENCE NO", "DRIVING LICENCE NO", "DL NO"],
        "permit_number": ["PERMIT NO", "PERMIT NUMBER"],
        "permit_type": ["PERMIT TYPE", "TYPE OF PERMIT"],
        "class": ["CLASS OF VEHICLE", "VEHICLE CLASS", "CLASS", "CATEGORY"],
        "nationality": ["NATIONALITY"],
        "dob": ["DATE OF BIRTH", "DOB", "BIRTH DATE"],
        "issue_date": ["DATE OF ISSUE", "ISSUE DATE", "ISSUED ON", "DATE OF ISSUANCE"],
        "expiry": ["DATE OF EXPIRY", "EXPIRY DATE", "EXPIRATION DATE", "VALID UNTIL", "VALID TO"],
        "sex": ["SEX", "GENDER"],
        "roll_number": ["STUDENT CODE", "STUDENT ID", "ROLL NO", "ROLL NUMBER", "REG NO", "REGISTRATION NO", "ENROLLMENT NO", "SEAT NO", "CODE"],
        "course": ["PROGRAMME", "PROGRAM", "COURSE", "BRANCH", "DEPARTMENT", "STREAM", "DEGREE"],
        "passing_year": ["PASSING YEAR", "YEAR OF PASSING", "EXAMINATION IN", "HELD IN", "SESSION"],
        "percentage": ["PERCENTAGE", "MARKS OBTAINED", "TOTAL MARKS", "DIVISION", "GRADE", "CGPA"],
    }

    for field in data:
        data[field] = value_after_label(field_labels.get(field, []))

    # Clean student code if it matched label
    if document_type in ["college_id", "marksheet"]:
        code_match = re.search(r"(?:STUDENT\s*CODE|STUDENT\s*ID|ROLL\s*NO\.?|ROLL\s*NUMBER|REG\s*NO\.?|ENROLLMENT\s*NO\.?)[:\s.-]*([A-Z0-9/]{4,25})", full_text)
        if code_match and code_match.group(1).upper() != "ISTRAR":
            data["roll_number"] = code_match.group(1).strip()
            
        prog_match = re.search(r"(?:PROGRAMME|PROGRAM|COURSE|BRANCH|STREAM)[:\s.-]*([A-Z0-9.()\s-]{3,30})(?:\s+SESSION|\s+VALID|\n|$)", full_text)
        if prog_match:
            data["course"] = prog_match.group(1).strip()

    # Specific regex extractors for Full Name
    name_label_match = re.search(r"(?:NAME|STUDENT\s*NAME|CANDIDATE\s*NAME|MR\./MS\.|MR\.|MS\.|SHRI|SMT\.)[:\s.-]*([A-Z\s]{3,40})(?:\s+STUDENT\s*CODE|\s+ROLL|\s+SON|\s+DAUGHTER|\s+S/O|\s+D/O|\s+HAVING|\s+BLOOD|\n|$)", full_text)
    if name_label_match:
        data["name"] = name_label_match.group(1).strip()

    if document_type == "national_id" and not data.get("id_number"):
        aadhaar = re.search(r"(?<!\d)(?:\d[ -]?){12}(?!\d)", full_text)
        if aadhaar:
            data["id_number"] = re.sub(r"[ -]", "", aadhaar.group(0))
    elif document_type == "national_id" and data.get("id_number"):
        data["id_number"] = re.sub(r"[ -]", "", data["id_number"])

    number_patterns = {
        "visa": ("visa_number", r"\b[A-Z]{0,2}\d{6,12}\b"),
        "driving_license": ("license_number", r"\b[A-Z]{1,4}[- ]?\d{4,16}\b"),
        "permit": ("permit_number", r"\b[A-Z]{0,4}[- ]?\d{5,16}\b"),
        "aadhaar": ("aadhaar_number", r"(?<!\d)(?:\d[ -]*){12}(?!\d)"),
        "pan_card": ("pan_number", r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"),
        "college_id": ("roll_number", r"\b(?:ROLL|ID|NO)[:\s]*([A-Z0-9-]{4,15})\b"),
        "marksheet": ("roll_number", r"\b(?:ROLL\s*NO\.?|ROLL)[:\s]*([0-9A-Z]{5,20})\b"),
        "voter_id": ("voter_id_number", r"\b[A-Z]{3}[0-9]{7}\b"),
    }
    if document_type in number_patterns and not data.get(number_patterns[document_type][0]):
        field, pattern = number_patterns[document_type]
        match = re.search(pattern, full_text)
        if match:
            data[field] = match.group(0).replace(" ", "")

    date_values = re.findall(
        r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b",
        full_text
    )
    date_count = len(date_values)
    if "dob" in data and not data["dob"] and date_values:
        data["dob"] = date_values.pop(0)
    if "issue_date" in data and not data["issue_date"] and date_count >= 2 and date_values:
        data["issue_date"] = date_values.pop(0)
    if "expiry" in data and not data["expiry"] and date_count >= 2 and date_values:
        data["expiry"] = date_values.pop(-1)

    if document_type == "passport":
        passport_data = parse_passport_text(text_lines)
        for field in data:
            if not data[field] and passport_data.get(field):
                data[field] = passport_data[field]

    return data


def parse_passport_text(text_lines: list) -> dict:

    data = {
        "name": None,
        "surname": None,
        "given_names": None,
        "passport_number": None,
        "dob": None,
        "expiry": None,
        "issue_date": None,
        "nationality": None,
        "sex": None,
        "place_of_birth": None,
        "place_of_issue": None
    }

    date_pattern = re.compile(
        r'\d{2}[-/]\d{2}[-/]\d{4}'
    )

    passport_pattern = re.compile(
        r'^(?=.*\d)[A-Z0-9]{7,9}$'
    )

    dates_found = []

    for i, line in enumerate(text_lines):

        line_clean = line.strip().upper()


        if '<' in line_clean and len(line_clean) > 10:
            continue

        # Dates
        dates = date_pattern.findall(line_clean)
        dates_found.extend(dates)

        # Passport number
        candidate = line_clean.replace(" ", "")

        if (
            passport_pattern.match(candidate)
            and not data["passport_number"]
        ):
            data["passport_number"] = candidate

        # Nationality
        if "INDIAN" in line_clean or re.search(r'\bIND\b', line_clean):
            data["nationality"] = "INDIAN"

        # Sex
        if re.fullmatch(r'[MF]', line_clean):
            data["sex"] = line_clean

        # Surname
        if "SURNAME" in line_clean and i + 1 < len(text_lines):
            data["surname"] = text_lines[i + 1].strip().upper()

        # Given names
        if (
            "GIVEN NAMES" in line_clean
            or "GIVEN NAME" in line_clean
        ):
            if i + 1 < len(text_lines):
                data["given_names"] = (
                    text_lines[i + 1].strip().upper()
                )

        # Place of birth
        if "PLACE OF BIRTH" in line_clean and i + 1 < len(text_lines):
            data["place_of_birth"] = (
                text_lines[i + 1].strip().upper()
            )

        # Place of issue
        if "PLACE OF ISSUE" in line_clean and i + 1 < len(text_lines):
            data["place_of_issue"] = (
                text_lines[i + 1].strip().upper()
            )

    # Dates
    if dates_found:
        if len(dates_found) >= 1:
            data["dob"] = dates_found[0]

        if len(dates_found) >= 2:
            data["issue_date"] = dates_found[1]

        if len(dates_found) >= 3:
            data["expiry"] = dates_found[-1]

    # Name
    if data["surname"] or data["given_names"]:
        data["name"] = " ".join(
            x for x in [
                data["given_names"],
                data["surname"]
            ] if x
        )

    return data