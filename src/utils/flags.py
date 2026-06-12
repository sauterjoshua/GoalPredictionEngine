"""Ländernamen → ISO-2-Code + Flaggen-Emoji. Single Source of Truth für alle Lieferschichten."""

COUNTRY_TO_ISO = {
    "Germany": "DE", "France": "FR", "Spain": "ES", "Italy": "IT",
    "England": "GB", "Brazil": "BR", "Argentina": "AR", "Portugal": "PT",
    "Netherlands": "NL", "Belgium": "BE", "Croatia": "HR", "USA": "US",
    "United States": "US", "Canada": "CA", "Mexico": "MX", "Japan": "JP",
    "South Korea": "KR", "Curaçao": "CW", "Curacao": "CW", "Ecuador": "EC",
    "Ivory Coast": "CI", "Morocco": "MA", "Senegal": "SN", "Ghana": "GH",
    "Uruguay": "UY", "Colombia": "CO", "Switzerland": "CH", "Denmark": "DK",
    "Poland": "PL", "Austria": "AT", "Australia": "AU", "Qatar": "QA",
    "Saudi Arabia": "SA", "Nigeria": "NG", "Cameroon": "CM", "Serbia": "RS",
    "Norway": "NO", "Sweden": "SE", "Turkey": "TR", "Egypt": "EG",
    "Czechia": "CZ", "Paraguay": "PY", "Scotland": "GB", "Haiti": "HT",
    "Bosnia-Herzegovina": "BA", "Congo DR": "CD", "Cape Verde Islands": "CV",
}


def flag_emoji(country: str) -> str:
    """Wandelt einen Ländernamen in ein Flaggen-Emoji (Fallback: weiße Flagge)."""
    iso = COUNTRY_TO_ISO.get(country)
    if not iso:
        return "🏳️"
    return "".join(chr(0x1F1E6 + (ord(c) - ord("A"))) for c in iso.upper())
