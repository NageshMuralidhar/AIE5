import asyncio
import json
import os
from datetime import datetime
from test_workflow import run_workflow
from workflow import create_workflow
from generate_test_dataset import GOLDEN_DATASET_DIR, validate_test_case
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def run_golden_tests():
    """Run tests using the golden dataset."""
    
    # Load the golden dataset
    dataset_path = os.path.join(GOLDEN_DATASET_DIR, "golden_dataset.json")
    if not os.path.exists(dataset_path):
        print("Golden dataset not found. Generating new dataset...")
        from generate_test_dataset import generate_golden_dataset
        generate_golden_dataset()
    
    with open(dataset_path, 'r') as f:
        golden_dataset = json.load(f)
    
    # Initialize workflow
    workflow = create_workflow(os.getenv("TAVILY_API_KEY"))
    
    # Store test results
    test_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_version": golden_dataset["metadata"]["version"]
        },
        "results": []
    }
    
    # Run tests for each test case
    for test_case in golden_dataset["test_cases"]:
        print(f"\nRunning test case: {test_case['input']['query']}")
        try:
            # Run the workflow
            result = await run_workflow(
                workflow,
                test_case["input"]["query"],
                agent_type=test_case["input"]["agent_type"],
                context=test_case["input"]["context"]
            )
            
            # Validate the results
            validation_result = validate_test_case(test_case, result)
            
            # Add results
            test_results["results"].append({
                "test_case_id": test_case["id"],
                "query": test_case["input"]["query"],
                "success": all(v["passed"] for v in validation_result["validations"]),
                "validation_results": validation_result,
                "workflow_output": result
            })
            
            # Print progress
            success = all(v["passed"] for v in validation_result["validations"])
            status = "✅ Passed" if success else "❌ Failed"
            print(f"{status} - {test_case['input']['query']}")
            
        except Exception as e:
            print(f"❌ Error running test case: {str(e)}")
            test_results["results"].append({
                "test_case_id": test_case["id"],
                "query": test_case["input"]["query"],
                "success": False,
                "error": str(e)
            })
    
    # Save test results
    results_dir = os.path.join(GOLDEN_DATASET_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"test_results_{timestamp}.json")
    
    with open(output_file, "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Print summary
    total_tests = len(test_results["results"])
    passed_tests = sum(1 for r in test_results["results"] if r.get("success", False))
    
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.2f}%")
    print("="*50)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧪 Running Golden Dataset Tests")
    print("="*50)
    
    try:
        asyncio.run(run_golden_tests())
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        raise 