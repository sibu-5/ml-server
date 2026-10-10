def detect_url_rules(features):
    detections = []

    if features["has_ip"]:
        detections.append(
            "URL uses an IP address instead of a domain"
        )

    if features["has_at_symbol"]:
        detections.append(
            "URL contains an @ symbol"
        )

    if features["has_punycode"]:
        detections.append(
            "Punycode/look-alike domain indicator detected"
        )

    if features["has_encoded_chars"]:
        detections.append(
            "Percent-encoded URL characters detected"
        )

    if features["has_suspicious_tld"]:
        detections.append(
            "Suspicious TLD indicator detected"
        )

    if features["suspicious_keyword_count"] >= 2:
        detections.append(
            "Multiple credential/account-related keywords detected"
        )

    return detections
