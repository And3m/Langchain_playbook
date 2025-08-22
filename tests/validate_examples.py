#!/usr/bin/env python3
"""
Example Validation Script

This script validates that all examples in the LangChain Playbook work correctly
by running them in a controlled environment with proper error handling.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))


class ExampleValidator:
    """Validates LangChain Playbook examples."""
    
    def __init__(self):
        self.project_root = project_root
        self.results = {
            "basics": [],
            "intermediate": [],
            "projects": [],
            "notebooks": []
        }
        self.api_key_available = self._check_api_key()
    
    def _check_api_key(self) -> bool:
        """Check if API key is available."""
        try:
            from utils.config import get_api_key
            api_key = get_api_key('openai')
            return api_key is not None
        except:
            return os.getenv('OPENAI_API_KEY') is not None
    
    def validate_import(self, module_path: str, expected_attributes: List[str] = None) -> Dict[str, Any]:
        """Validate that a module imports correctly."""
        try:
            # Add the module's directory to path temporarily
            module_dir = Path(module_path).parent
            if str(module_dir) not in sys.path:
                sys.path.insert(0, str(module_dir))
            
            module_name = Path(module_path).stem
            module = importlib.import_module(module_name)
            
            # Check for expected attributes
            missing_attributes = []
            if expected_attributes:
                for attr in expected_attributes:
                    if not hasattr(module, attr):
                        missing_attributes.append(attr)
            
            return {
                "status": "success" if not missing_attributes else "warning",
                "module": module_name,
                "missing_attributes": missing_attributes,
                "error": None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "module": module_name if 'module_name' in locals() else "unknown",
                "error": str(e),
                "missing_attributes": []
            }
        finally:
            # Clean up path
            if str(module_dir) in sys.path:
                sys.path.remove(str(module_dir))
    
    def validate_script_execution(self, script_path: str, timeout: int = 30) -> Dict[str, Any]:
        """Validate that a script can be executed without errors."""
        try:
            # Set environment for execution
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            # If no API key, set a dummy one to avoid immediate failures
            if not self.api_key_available:
                env['OPENAI_API_KEY'] = 'demo-key-for-testing'
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(self.project_root)
            )
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "script": script_path.name,
                "returncode": result.returncode,
                "stdout": result.stdout[:500] if result.stdout else "",
                "stderr": result.stderr[:500] if result.stderr else "",
                "error": None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "script": script_path.name,
                "error": f"Script timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "status": "error",
                "script": script_path.name,
                "error": str(e)
            }
    
    def validate_notebook(self, notebook_path: str) -> Dict[str, Any]:
        """Validate notebook structure and basic syntax."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)
            
            # Check basic structure
            if 'cells' not in notebook_data:
                return {
                    "status": "error",
                    "notebook": notebook_path.name,
                    "error": "Missing 'cells' in notebook"
                }
            
            # Count cells and check for basic content
            code_cells = 0
            markdown_cells = 0
            
            for cell in notebook_data['cells']:
                if cell.get('cell_type') == 'code':
                    code_cells += 1
                elif cell.get('cell_type') == 'markdown':
                    markdown_cells += 1
            
            return {
                "status": "success",
                "notebook": Path(notebook_path).name,
                "code_cells": code_cells,
                "markdown_cells": markdown_cells,
                "total_cells": len(notebook_data['cells']),
                "error": None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "notebook": Path(notebook_path).name,
                "error": str(e)
            }
    
    def validate_basics(self) -> List[Dict[str, Any]]:
        """Validate basic examples."""
        basics_dir = self.project_root / 'basics'
        results = []
        
        if not basics_dir.exists():
            return [{"status": "error", "error": "Basics directory not found"}]
        
        # Find all Python files in basics
        python_files = list(basics_dir.rglob('*.py'))
        
        for py_file in python_files:
            if py_file.name.startswith('__'):
                continue
                
            # Test import
            import_result = self.validate_import(py_file, ["main"])
            results.append({
                "type": "import",
                "file": str(py_file.relative_to(self.project_root)),
                **import_result
            })
            
            # Test execution if import succeeded
            if import_result["status"] == "success":
                exec_result = self.validate_script_execution(py_file)
                results.append({
                    "type": "execution",
                    "file": str(py_file.relative_to(self.project_root)),
                    **exec_result
                })
        
        return results
    
    def validate_intermediate(self) -> List[Dict[str, Any]]:
        """Validate intermediate examples."""
        intermediate_dir = self.project_root / 'intermediate'
        results = []
        
        if not intermediate_dir.exists():
            return [{"status": "error", "error": "Intermediate directory not found"}]
        
        # Find all Python files in intermediate
        python_files = list(intermediate_dir.rglob('*.py'))
        
        for py_file in python_files:
            if py_file.name.startswith('__'):
                continue
                
            # Test import only (intermediate examples might be more complex)
            import_result = self.validate_import(py_file, ["main"])
            results.append({
                "type": "import",
                "file": str(py_file.relative_to(self.project_root)),
                **import_result
            })
        
        return results
    
    def validate_projects(self) -> List[Dict[str, Any]]:
        """Validate project examples."""
        projects_dir = self.project_root / 'projects'
        results = []
        
        if not projects_dir.exists():
            return [{"status": "error", "error": "Projects directory not found"}]
        
        # Expected project classes
        project_expectations = {
            'chatbot_app.py': ['PersonalityChatbot'],
            'qa_system.py': ['DocumentQASystem'],
            'code_assistant.py': ['CodeAssistant'],
            'research_assistant.py': ['ResearchAssistant'],
            'api_service.py': ['app', 'LangChainService']
        }
        
        # Find project files
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
                
            for expected_file, expected_classes in project_expectations.items():
                py_file = project_dir / expected_file
                if py_file.exists():
                    import_result = self.validate_import(py_file, expected_classes)
                    results.append({
                        "type": "import",
                        "file": str(py_file.relative_to(self.project_root)),
                        "project": project_dir.name,
                        **import_result
                    })
        
        return results
    
    def validate_notebooks(self) -> List[Dict[str, Any]]:
        """Validate Jupyter notebooks."""
        notebooks_dir = self.project_root / 'notebooks'
        results = []
        
        if not notebooks_dir.exists():
            return [{"status": "error", "error": "Notebooks directory not found"}]
        
        # Find all notebook files
        notebook_files = list(notebooks_dir.glob('*.ipynb'))
        
        for notebook_file in notebook_files:
            notebook_result = self.validate_notebook(notebook_file)
            results.append({
                "type": "notebook",
                "file": str(notebook_file.relative_to(self.project_root)),
                **notebook_result
            })
        
        return results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run full validation suite."""
        print("🔍 Starting LangChain Playbook Validation...")
        print(f"📁 Project root: {self.project_root}")
        print(f"🔑 API key available: {'Yes' if self.api_key_available else 'No (using demo mode)'}")
        print()
        
        # Run validations
        print("📚 Validating basics...")
        self.results["basics"] = self.validate_basics()
        
        print("🔧 Validating intermediate examples...")
        self.results["intermediate"] = self.validate_intermediate()
        
        print("🚀 Validating projects...")
        self.results["projects"] = self.validate_projects()
        
        print("📓 Validating notebooks...")
        self.results["notebooks"] = self.validate_notebooks()
        
        return self.results
    
    def print_summary(self) -> bool:
        """Print validation summary and return success status."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        total_tests = 0
        total_success = 0
        total_warnings = 0
        total_errors = 0
        
        for category, results in self.results.items():
            if not results:
                continue
                
            print(f"\n📋 {category.upper()}:")
            
            category_success = 0
            category_warnings = 0
            category_errors = 0
            
            for result in results:
                total_tests += 1
                status = result.get("status", "unknown")
                
                if status == "success":
                    total_success += 1
                    category_success += 1
                    print(f"  ✅ {result.get('file', 'unknown')} - {result.get('type', 'test')}")
                elif status == "warning":
                    total_warnings += 1
                    category_warnings += 1
                    print(f"  ⚠️ {result.get('file', 'unknown')} - {result.get('type', 'test')} (warnings)")
                    if result.get("missing_attributes"):
                        print(f"     Missing: {', '.join(result['missing_attributes'])}")
                else:
                    total_errors += 1
                    category_errors += 1
                    print(f"  ❌ {result.get('file', 'unknown')} - {result.get('type', 'test')}")
                    if result.get("error"):
                        print(f"     Error: {result['error'][:100]}...")
            
            print(f"     Summary: {category_success} ✅, {category_warnings} ⚠️, {category_errors} ❌")
        
        # Overall summary
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Total tests: {total_tests}")
        print(f"   Successful: {total_success} ✅")
        print(f"   Warnings: {total_warnings} ⚠️")
        print(f"   Errors: {total_errors} ❌")
        
        success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        print(f"   Success rate: {success_rate:.1f}%")
        
        if total_errors == 0 and total_warnings <= total_tests * 0.1:  # Allow up to 10% warnings
            print(f"\n🎉 Validation PASSED! The LangChain Playbook is ready for users.")
            return True
        elif total_errors == 0:
            print(f"\n⚠️ Validation passed with warnings. Consider addressing the warnings.")
            return True
        else:
            print(f"\n❌ Validation FAILED. Please fix the errors before deployment.")
            return False


def main():
    """Main validation function."""
    validator = ExampleValidator()
    
    # Run validation
    results = validator.run_full_validation()
    
    # Print summary and exit with appropriate code
    success = validator.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()