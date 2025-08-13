"""
Basic tests for regional measurement preferences.

Tests the core functionality of the regional measurement system to ensure
regional defaults work correctly without extensive test infrastructure.
"""
import pytest
from stt.text_formatting.constants import get_regional_defaults, get_measurement_preference
from stt.text_formatting.formatter import TextFormatter


class TestRegionalDefaults:
    """Test regional measurement defaults and fallback logic."""

    def test_supported_regions(self):
        """Test that supported regions return correct defaults."""
        # US uses Imperial/Fahrenheit
        us_defaults = get_regional_defaults("en-US")
        assert us_defaults["temperature"] == "fahrenheit"
        assert us_defaults["length"] == "imperial"
        assert us_defaults["weight"] == "imperial"
        
        # UK uses Metric/Celsius
        uk_defaults = get_regional_defaults("en-GB")
        assert uk_defaults["temperature"] == "celsius"
        assert uk_defaults["length"] == "metric"
        assert uk_defaults["weight"] == "metric"
        
        # Canada uses Metric/Celsius
        ca_defaults = get_regional_defaults("en-CA")
        assert ca_defaults["temperature"] == "celsius"
        assert ca_defaults["length"] == "metric"
        assert ca_defaults["weight"] == "metric"

    def test_unsupported_english_variants_fallback(self):
        """Test that unsupported English variants fall back to metric."""
        # Australia should use metric (not US imperial)
        au_defaults = get_regional_defaults("en-AU")
        assert au_defaults["temperature"] == "celsius"
        assert au_defaults["length"] == "metric"
        assert au_defaults["weight"] == "metric"
        
        # India should use metric  
        in_defaults = get_regional_defaults("en-IN")
        assert in_defaults["temperature"] == "celsius"
        assert in_defaults["length"] == "metric"
        assert in_defaults["weight"] == "metric"

    def test_non_english_languages_fallback(self):
        """Test that non-English languages fall back to metric."""
        # German should use metric
        de_defaults = get_regional_defaults("de")
        assert de_defaults["temperature"] == "celsius"
        assert de_defaults["length"] == "metric"
        assert de_defaults["weight"] == "metric"
        
        # French should use metric (has fr.json but no regional_defaults)
        fr_defaults = get_regional_defaults("fr")
        assert fr_defaults["temperature"] == "celsius"
        assert fr_defaults["length"] == "metric"
        assert fr_defaults["weight"] == "metric"

    def test_measurement_preference_function(self):
        """Test the measurement preference function with overrides."""
        # Test user override takes precedence
        assert get_measurement_preference("en-US", "temperature", "celsius") == "celsius"
        assert get_measurement_preference("en-GB", "temperature", "fahrenheit") == "fahrenheit"
        
        # Test regional defaults when no override
        assert get_measurement_preference("en-US", "temperature") == "fahrenheit"
        assert get_measurement_preference("en-GB", "temperature") == "celsius"
        assert get_measurement_preference("en-AU", "temperature") == "celsius"
        
        # Test invalid override falls back to regional default
        assert get_measurement_preference("en-US", "temperature", "kelvin") == "fahrenheit"
        assert get_measurement_preference("en-GB", "temperature", "invalid") == "celsius"
        
        # Test invalid measurement type falls back to metric
        assert get_measurement_preference("en-US", "pressure") == "metric"


class TestRegionalTemperatureFormatting:
    """Test actual temperature formatting with regional preferences."""

    def test_temperature_formatting_by_region(self):
        """Test that ambiguous temperatures use correct regional defaults."""
        test_text = "the temperature is twenty degrees"
        
        # US should default to Fahrenheit
        us_formatter = TextFormatter(language="en-US", regional_config={})
        us_result = us_formatter.format_transcription(test_text)
        assert "20°F" in us_result
        
        # UK should default to Celsius
        uk_formatter = TextFormatter(language="en-GB", regional_config={})
        uk_result = uk_formatter.format_transcription(test_text)
        assert "20°C" in uk_result
        
        # Australia should default to Celsius (not Fahrenheit)
        au_formatter = TextFormatter(language="en-AU", regional_config={})
        au_result = au_formatter.format_transcription(test_text)
        assert "20°C" in au_result

    def test_explicit_units_preserved(self):
        """Test that explicit temperature units are always preserved."""
        us_formatter = TextFormatter(language="en-US", regional_config={})
        uk_formatter = TextFormatter(language="en-GB", regional_config={})
        
        # Explicit Celsius should be preserved regardless of region
        celsius_text = "twenty degrees celsius"
        assert "20°C" in us_formatter.format_transcription(celsius_text)
        assert "20°C" in uk_formatter.format_transcription(celsius_text)
        
        # Explicit Fahrenheit should be preserved regardless of region
        fahrenheit_text = "twenty degrees fahrenheit"
        assert "20°F" in us_formatter.format_transcription(fahrenheit_text)
        assert "20°F" in uk_formatter.format_transcription(fahrenheit_text)

    def test_user_overrides(self):
        """Test that user configuration overrides work correctly."""
        test_text = "the temperature is twenty degrees"
        
        # US user overriding to Celsius
        us_celsius_config = {"unit_temperature": "celsius"}
        us_formatter = TextFormatter(language="en-US", regional_config=us_celsius_config)
        result = us_formatter.format_transcription(test_text)
        assert "20°C" in result
        
        # UK user overriding to Fahrenheit
        uk_fahrenheit_config = {"unit_temperature": "fahrenheit"}
        uk_formatter = TextFormatter(language="en-GB", regional_config=uk_fahrenheit_config)
        result = uk_formatter.format_transcription(test_text)
        assert "20°F" in result