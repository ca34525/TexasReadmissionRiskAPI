# src/utils.py

"""
Utility functions for parsing specific, nested elements within FHIR JSON resources.
These helpers provide a safe and reusable way to extract data without cluttering
the main ETL parsing logic.
"""


def get_clean_id(ref_dict: dict) -> str | None:
    """Extracts and robustly cleans the ID from a raw FHIR reference dictionary."""
    if not (ref_dict and isinstance(ref_dict, dict) and "reference" in ref_dict):
        return None
    ref_string = ref_dict["reference"]
    last_delim_pos = max(ref_string.rfind("/"), ref_string.rfind(":"))
    return ref_string[last_delim_pos + 1 :] if last_delim_pos != -1 else ref_string


def get_extension_value(resource: dict, url: str) -> str | float | None:
    """Finds a specific extension by URL and returns its value."""
    if not resource or not isinstance(resource, dict):
        return None
    for ext in resource.get("extension", []):
        if ext.get("url") == url:
            for key in [
                "valueDecimal",
                "valueString",
                "valueDate",
                "valueCode",
                "valueInteger",
            ]:
                if key in ext:
                    return ext[key]
            if "valueMoney" in ext and "value" in ext["valueMoney"]:
                return ext["valueMoney"]["value"]
    return None


def get_identifier(resource: dict, system_url: str) -> str | None:
    """Finds a specific identifier by its system URL."""
    for identifier in resource.get("identifier", []):
        if identifier.get("system") == system_url:
            return identifier.get("value")
    return None


def get_first_address(resource: dict) -> dict | None:
    """Safely gets the first address entry from a resource."""
    if addresses := resource.get("address"):
        return addresses[0]
    return None


def get_coding(codeable_concept: dict) -> dict:
    """Extracts the first coding from a CodeableConcept."""
    if codeable_concept and (codings := codeable_concept.get("coding")):
        return codings[0]
    return {}


def get_us_core_extension_value(resource: dict, url: str) -> str | None:
    """
    Specifically parses the complex US Core Race & Ethnicity extensions,
    which have a nested valueCoding structure.
    """
    if not resource or not isinstance(resource, dict):
        return None
    for ext in resource.get("extension", []):
        if ext.get("url") == url:
            # The value is inside another 'extension' list
            nested_ext = next(iter(ext.get("extension", [])), None)
            if nested_ext and "valueCoding" in nested_ext:
                return nested_ext["valueCoding"].get("display")
    return None
