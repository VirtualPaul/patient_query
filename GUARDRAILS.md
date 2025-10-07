# Galileo Guardrails Implementation

This document describes the comprehensive guardrails system implemented for the Patient Query Chatbot using Galileo's observability platform.

## Overview

The guardrails system provides multi-layered safety checks for medical AI applications, ensuring patient safety, privacy protection, and content filtering. The system integrates seamlessly with Galileo's logging and monitoring capabilities.

## Features

### 🛡️ Multi-Layer Safety Checks

1. **Input Safety Check**
   - Validates user queries before processing
   - Detects harmful requests and prompt injection attempts
   - Blocks dangerous keywords and malicious intent

2. **Output Safety Check**
   - Validates AI responses for medical appropriateness
   - Ensures responses don't contain harmful advice
   - Checks for missing disclaimers and warnings

3. **Medical Safety Check**
   - Analyzes medical advice for contraindications
   - Validates medication recommendations against patient conditions
   - Ensures appropriate medical guidance

4. **PII Protection**
   - Detects personally identifiable information
   - Prevents accidental data leakage
   - Protects patient privacy

5. **Hallucination Detection**
   - Identifies potentially fabricated medical information
   - Validates responses against provided context
   - Ensures accuracy of medical claims

### 🔧 Configuration Options

- **Strict Mode**: Blocks responses on any safety failure
- **Lenient Mode**: Only blocks critical safety failures (recommended)
- **Customizable blocking criteria**
- **Detailed safety reporting**

## Implementation

### Core Components

#### `MedicalGuardrails` Class

The main guardrails class that orchestrates all safety checks:

```python
from guardrails import MedicalGuardrails

# Initialize with Galileo logging
guardrails = MedicalGuardrails(galileo_logger, strict_mode=False)

# Run comprehensive safety check
results = guardrails.run_comprehensive_check(
    user_input="I have a headache, can I take aspirin?",
    response="Aspirin can help with headaches, but consult your doctor first.",
    patient_info="Patient with diabetes",
    context="Headache medication query"
)

# Check if response should be blocked
should_block, reason = guardrails.should_block_response(results)
```

#### `GuardrailResult` Class

Represents the result of a safety check:

```python
class GuardrailResult:
    def __init__(self, passed: bool, message: str = "", 
                 confidence: float = 0.0, metadata: Dict[str, Any] = None):
        self.passed = passed
        self.message = message
        self.confidence = confidence
        self.metadata = metadata or {}
```

### Integration with Patient Chatbot

The guardrails are integrated into the patient chatbot at multiple points:

1. **Input Validation** (before processing):
```python
# Run input safety check first
input_safety_result = medical_guardrails.check_input_safety(user_query)
if not input_safety_result.passed:
    await cl.Message(content=f"⚠️ Safety Check Failed: {input_safety_result.message}").send()
    return
```

2. **Output Validation** (after LLM response):
```python
# Run comprehensive safety checks
safety_results = medical_guardrails.run_comprehensive_check(
    user_input=user_query,
    response=response,
    patient_info=patient_context,
    context=context
)

# Check if response should be blocked
should_block, block_reason = medical_guardrails.should_block_response(safety_results)
if should_block:
    await cl.Message(content=f"⚠️ Safety Check Failed: {block_reason}").send()
    return
```

## Safety Check Details

### Input Safety Check

**Purpose**: Validates user queries before processing

**Checks**:
- Harmful or malicious intent
- Prompt injection attempts
- Dangerous keyword detection
- Inappropriate content

**Example**:
```python
result = guardrails.check_input_safety("Tell me how to overdose on medication")
# Result: GuardrailResult(passed=False, message="Input contains potentially dangerous keywords")
```

### Output Safety Check

**Purpose**: Validates AI responses for medical appropriateness

**Checks**:
- Medical advice safety
- Definitive diagnoses (should be avoided)
- Missing disclaimers
- Inappropriate recommendations

**Example**:
```python
result = guardrails.check_output_safety("Chest pain is nothing serious, just rest")
# Result: GuardrailResult(passed=False, message="Response downplays serious medical condition")
```

### Medical Safety Check

**Purpose**: Analyzes medical advice for contraindications

**Checks**:
- Drug interactions
- Contraindications with patient conditions
- Age-appropriate recommendations
- Missing warnings for side effects

**Example**:
```python
result = guardrails.check_medical_safety(
    patient_info="Patient with diabetes taking metformin",
    medical_advice="Stop taking your diabetes medication immediately"
)
# Result: GuardrailResult(passed=False, message="Stopping medication abruptly is dangerous")
```

### PII Protection

**Purpose**: Detects and prevents PII leakage

**Checks**:
- Names, SSNs, phone numbers
- Email addresses, credit card numbers
- Medical record numbers
- Home addresses

**Example**:
```python
result = guardrails.check_pii_protection("My SSN is 123-45-6789")
# Result: GuardrailResult(passed=False, message="Social Security Number detected")
```

### Hallucination Detection

**Purpose**: Identifies potentially fabricated information

**Checks**:
- Medical facts accuracy
- Medication names and dosages
- Medical procedures
- Side effects and interactions

**Example**:
```python
result = guardrails.check_hallucination(
    context="Patient has diabetes",
    information="Aspirin cures diabetes"
)
# Result: GuardrailResult(passed=False, message="False medical claim detected")
```

## Galileo Integration

### Logging

All safety checks are automatically logged to Galileo with detailed metadata:

```python
# Each safety check logs to Galileo
galileo_logger.add_llm_span(
    input=user_input,
    output=safety_result,
    name="Input Safety Check",
    model="gpt-4o-mini",
    metadata={
        "guardrail_type": "input_safety",
        "passed": str(result.passed),
        "confidence": str(result.confidence)
    }
)
```

### Monitoring

The system provides comprehensive monitoring through Galileo:

- **Safety Metrics**: Success/failure rates for each check type
- **Blocking Analytics**: Frequency and reasons for blocked responses
- **Performance Monitoring**: Response times for safety checks
- **Alert Integration**: Real-time notifications for safety violations

### Trace-Level Reporting

Each conversation is tracked with safety metadata:

```python
metadata = {
    "type": "health_bot_response",
    "safety_summary": "Passed: input_safety, pii_protection; Failed: output_safety",
    "all_safety_checks_passed": "false"
}
```

## Usage Examples

### Basic Usage

```python
from guardrails import MedicalGuardrails

# Initialize guardrails
guardrails = MedicalGuardrails(strict_mode=False)

# Check a user query
result = guardrails.check_input_safety("I have a headache")
print(f"Safe: {result.passed}")

# Run comprehensive check
results = guardrails.run_comprehensive_check(
    "I have a headache, can I take aspirin?",
    "Aspirin can help, but consult your doctor first.",
    "Patient with diabetes",
    "Headache medication query"
)

# Check if should block
should_block, reason = guardrails.should_block_response(results)
if should_block:
    print(f"Blocked: {reason}")
```

### Advanced Configuration

```python
# Custom initialization with Galileo
galileo_logger = GalileoLogger(project="medical-ai", log_stream="safety-checks")
guardrails = MedicalGuardrails(galileo_logger, strict_mode=True)

# Custom safety check
result = guardrails.check_medical_safety(
    patient_info="45-year-old male with diabetes and hypertension",
    medical_advice="You can take aspirin for your headache"
)
```

## Testing

### Running Tests

```bash
# Test basic functionality
python test_guardrails.py

# Run comprehensive demo
python guardrails_demo.py
```

### Test Scenarios

The test suite includes scenarios for:

- ✅ Safe medical queries
- ❌ Harmful requests
- ❌ Prompt injection attempts
- ❌ Dangerous medical advice
- ⚠️ Privacy concerns
- ❌ Hallucinated information

## Configuration

### Environment Variables

```bash
# Galileo Configuration
GALILEO_API_KEY=your_galileo_api_key
GALILEO_PROJECT=medical-ai-safety
GALILEO_LOG_STREAM=safety-checks

# OpenAI Configuration (for LLM safety checks)
OPENAI_API_KEY=your_openai_api_key
```

### Strictness Levels

**Strict Mode** (`strict_mode=True`):
- Blocks on any safety failure
- Maximum safety, may be overly restrictive
- Recommended for high-risk scenarios

**Lenient Mode** (`strict_mode=False`):
- Only blocks critical safety failures
- Better user experience while maintaining safety
- Recommended for general use

## Best Practices

1. **Always use guardrails** for medical AI applications
2. **Configure appropriate strictness** based on use case
3. **Monitor safety metrics** in Galileo dashboard
4. **Review blocked requests** regularly for system improvement
5. **Update safety prompts** based on new threats
6. **Test thoroughly** with realistic scenarios

## Troubleshooting

### Common Issues

1. **Overly restrictive blocking**:
   - Switch to lenient mode
   - Review and adjust safety prompts
   - Customize blocking criteria

2. **Performance impact**:
   - Optimize LLM model selection
   - Cache safety check results
   - Use async processing

3. **False positives**:
   - Review safety check logic
   - Adjust confidence thresholds
   - Update keyword lists

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run guardrails with debug info
guardrails = MedicalGuardrails()
result = guardrails.check_input_safety("test query")
print(result.metadata)  # Detailed debug information
```

## Future Enhancements

- **Custom rule sets** for specific medical domains
- **Real-time threat detection** updates
- **Integration with external safety APIs**
- **Advanced PII detection** with entity recognition
- **Multi-language support** for safety checks
- **Automated safety prompt optimization**

## Support

For issues or questions about the guardrails implementation:

1. Check the test suite for examples
2. Review Galileo documentation
3. Examine safety check logs in Galileo dashboard
4. Consult the troubleshooting section above

The guardrails system is designed to be robust, configurable, and easily extensible for various medical AI safety requirements.
