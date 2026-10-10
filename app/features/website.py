"""
Future website/HTML feature service.

This should eventually handle controlled webpage inspection,
login-form detection, iframe analysis, redirects, etc.

Do not fetch arbitrary user URLs directly from the FastAPI process.
Use an isolated, SSRF-protected worker when implemented.
"""
