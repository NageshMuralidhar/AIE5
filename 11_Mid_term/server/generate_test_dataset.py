import json
import os
from uuid import uuid4
from datetime import datetime

# Define the path for the golden dataset
GOLDEN_DATASET_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(GOLDEN_DATASET_DIR, exist_ok=True)

def create_test_case(query, agent_type="believer", context=None):
    """Helper function to create a test case with standard structure."""
    return {
        "id": str(uuid4()),
        "input": {
            "query": query,
            "agent_type": agent_type,
            "context": context
        },
        "expected_output": {
            "debate_structure": {
                "min_turns": 2,
                "max_turns": 4,
                "required_agents": ["extractor", "believer", "skeptic", "supervisor"]
            },
            "transcript_requirements": {
                "required_fields": ["id", "podcastScript", "topic"],
                "topic_match": True,
                "min_script_length": 500
            },
            "podcast_requirements": {
                "required_fields": ["content", "audio_file", "title", "description", "category"],
                "audio_format": "mp3",
                "min_file_size": 1000
            }
        }
    }

def generate_test_cases():
    """Generate 50 diverse test cases."""
    test_cases = []
    
    # Technology Topics (10 cases)
    tech_topics = [
        ("Are electric vehicles better for the environment?", "believer"),
        ("Should artificial intelligence be regulated?", "skeptic"),
        ("Is blockchain technology revolutionizing finance?", "believer"),
        ("Are smart homes making us too dependent on technology?", "skeptic"),
        ("Should social media platforms be responsible for content moderation?", "believer"),
        ("Is 5G technology safe for public health?", "skeptic"),
        ("Will quantum computing make current encryption obsolete?", "believer"),
        ("Should facial recognition be used in public spaces?", "skeptic"),
        ("Are autonomous vehicles ready for widespread adoption?", "believer"),
        ("Does virtual reality have practical applications beyond gaming?", "skeptic")
    ]
    
    # Society and Culture (10 cases)
    society_topics = [
        ("Is remote work the future of employment?", "believer"),
        ("Should universal basic income be implemented globally?", "skeptic"),
        ("Are social media platforms harming mental health?", "believer"),
        ("Should voting be mandatory?", "skeptic"),
        ("Is cancel culture beneficial for society?", "believer"),
        ("Should there be limits on free speech online?", "skeptic"),
        ("Are gender quotas effective in achieving equality?", "believer"),
        ("Should religious education be part of public schools?", "skeptic"),
        ("Is multiculturalism strengthening or weakening societies?", "believer"),
        ("Should citizenship be available for purchase?", "skeptic")
    ]
    
    # Environment and Sustainability (10 cases)
    environment_topics = [
        ("Can renewable energy completely replace fossil fuels?", "believer"),
        ("Should nuclear power be part of climate change solution?", "skeptic"),
        ("Is carbon pricing effective in reducing emissions?", "believer"),
        ("Should single-use plastics be completely banned?", "skeptic"),
        ("Are vertical farms the future of agriculture?", "believer"),
        ("Should meat consumption be regulated for environmental reasons?", "skeptic"),
        ("Is geoengineering a viable solution to climate change?", "believer"),
        ("Should private companies be allowed to exploit space resources?", "skeptic"),
        ("Are carbon offsets an effective environmental solution?", "believer"),
        ("Should environmental protection override economic growth?", "skeptic")
    ]
    
    # Health and Wellness (10 cases)
    health_topics = [
        ("Should healthcare be completely free?", "believer"),
        ("Is telemedicine as effective as traditional healthcare?", "skeptic"),
        ("Should vaccines be mandatory?", "believer"),
        ("Is genetic engineering of humans ethical?", "skeptic"),
        ("Should alternative medicine be covered by insurance?", "believer"),
        ("Is human enhancement technology ethical?", "skeptic"),
        ("Should organ donation be opt-out rather than opt-in?", "believer"),
        ("Are fitness trackers improving public health?", "skeptic"),
        ("Should sugar be regulated like tobacco?", "believer"),
        ("Is meditation effective as mental health treatment?", "skeptic")
    ]
    
    # Education and Career (10 cases)
    education_topics = [
        ("Should college education be free?", "believer"),
        ("Is standardized testing effective?", "skeptic"),
        ("Should coding be mandatory in schools?", "believer"),
        ("Are traditional degrees becoming obsolete?", "skeptic"),
        ("Should student debt be forgiven?", "believer"),
        ("Is homeschooling as effective as traditional schooling?", "skeptic"),
        ("Should arts education be mandatory?", "believer"),
        ("Are gap years beneficial for students?", "skeptic"),
        ("Should schools teach financial literacy?", "believer"),
        ("Is year-round schooling better for learning?", "skeptic")
    ]
    
    # Add all topics to test cases
    for topics in [tech_topics, society_topics, environment_topics, health_topics, education_topics]:
        for query, agent_type in topics:
            test_cases.append(create_test_case(query, agent_type))
    
    return test_cases

def generate_golden_dataset():
    """Generate a golden dataset for testing the podcast debate system."""
    
    # Get test cases
    test_cases = generate_test_cases()
    
    # Create sample transcripts
    sample_transcripts = [
        {
            "id": str(uuid4()),
            "podcastScript": """**Podcast Script: Electric Vehicles and the Environment**

Host: Welcome to our debate on the environmental impact of electric vehicles...

Skeptic: While EVs reduce direct emissions, we must consider the environmental cost of battery production...

Believer: The long-term benefits of EVs in reducing carbon emissions far outweigh the initial production impact...""",
            "topic": "Are electric vehicles better for the environment?"
        },
        {
            "id": str(uuid4()),
            "podcastScript": """**Podcast Script: AI Regulation Debate**

Host: Today we're exploring the complex topic of AI regulation...

Skeptic: Without proper oversight, AI development could lead to serious societal risks...

Believer: Smart regulation can help us harness AI's benefits while minimizing potential harm...""",
            "topic": "Should artificial intelligence be regulated?"
        }
    ]

    # Create the golden dataset structure
    golden_dataset = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "description": "Golden dataset for testing the podcast debate system",
            "total_test_cases": len(test_cases),
            "categories": [
                "Technology",
                "Society and Culture",
                "Environment and Sustainability",
                "Health and Wellness",
                "Education and Career"
            ]
        },
        "test_cases": test_cases,
        "sample_transcripts": sample_transcripts,
        "validation_rules": {
            "debate": {
                "required_agents": ["extractor", "believer", "skeptic", "supervisor"],
                "min_debate_turns": 2,
                "max_debate_turns": 4
            },
            "transcript": {
                "required_fields": ["id", "podcastScript", "topic"],
                "min_script_length": 500
            },
            "podcast": {
                "required_fields": ["content", "audio_file", "title", "description", "category"],
                "supported_audio_formats": ["mp3"],
                "min_file_size": 1000
            }
        }
    }

    # Save the golden dataset
    output_file = os.path.join(GOLDEN_DATASET_DIR, "golden_dataset.json")
    with open(output_file, "w") as f:
        json.dump(golden_dataset, f, indent=2)

    print(f"Golden dataset generated successfully at: {output_file}")
    return golden_dataset

def validate_test_case(test_case, actual_output):
    """Validate a test case against actual output."""
    validation_results = {
        "test_case_id": test_case["id"],
        "query": test_case["input"]["query"],
        "validations": []
    }

    # Validate debate structure
    expected_structure = test_case["expected_output"]["debate_structure"]
    debate_history = actual_output.get("debate_history", [])
    
    validation_results["validations"].append({
        "check": "debate_turns",
        "passed": expected_structure["min_turns"] <= len(debate_history) <= expected_structure["max_turns"],
        "details": f"Expected {expected_structure['min_turns']}-{expected_structure['max_turns']} turns, got {len(debate_history)}"
    })

    # Validate transcript
    transcript_reqs = test_case["expected_output"]["transcript_requirements"]
    if "transcript" in actual_output:
        transcript = actual_output["transcript"]
        validation_results["validations"].append({
            "check": "transcript_fields",
            "passed": all(field in transcript for field in transcript_reqs["required_fields"]),
            "details": "Transcript field validation"
        })

    # Validate podcast output
    podcast_reqs = test_case["expected_output"]["podcast_requirements"]
    if "final_podcast" in actual_output:
        podcast = actual_output["final_podcast"]
        validation_results["validations"].append({
            "check": "podcast_fields",
            "passed": all(field in podcast for field in podcast_reqs["required_fields"]),
            "details": "Podcast field validation"
        })

    return validation_results

if __name__ == "__main__":
    # Generate the golden dataset
    dataset = generate_golden_dataset()
    print("\nGolden Dataset Summary:")
    print(f"Number of test cases: {len(dataset['test_cases'])}")
    print(f"Number of sample transcripts: {len(dataset['sample_transcripts'])}")
    print(f"Categories covered: {dataset['metadata']['categories']}")
    print(f"Validation rules defined: {list(dataset['validation_rules'].keys())}") 