#!/bin/bash
# Run all tests for the Unified Privacy Pipeline

echo "========================================="
echo "Running Unified Privacy Pipeline Tests"
echo "========================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run pytest with coverage
echo "Running tests with coverage..."
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    echo "Coverage report generated in htmlcov/index.html"
else
    echo ""
    echo "❌ Some tests failed. Please review the output above."
    exit 1
fi
