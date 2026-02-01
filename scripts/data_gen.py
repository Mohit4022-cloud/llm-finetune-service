#!/usr/bin/env python3
"""
Generate training data for Enterprise-to-Casual LLM fine-tuning.
Creates 50 diverse examples of corporate emails with their casual Slack equivalents.
"""

import json
import os
from typing import List, Dict

# Training data templates - 10 categories with 5 variations each
TEMPLATES = [
    # Category 1: Status Updates (5 examples)
    {
        "input": "I am writing to inform you that the project timeline has been extended by two weeks due to unforeseen technical complications. We apologize for any inconvenience this may cause.",
        "output": "Hey! Quick update: we're pushing the project timeline by 2 weeks due to some tech issues. Sorry for the delay! 😅"
    },
    {
        "input": "I am writing to inform you that the quarterly report has been finalized and is now available for review in the shared drive.",
        "output": "Hey! Quick update: Q4 report is done and ready in the shared drive 📊"
    },
    {
        "input": "I am writing to inform you that the server maintenance has been completed successfully and all systems are operational.",
        "output": "Hey! Quick update: server maintenance is done, everything's back online ✅"
    },
    {
        "input": "I am writing to inform you that we have secured additional budget for the marketing campaign.",
        "output": "Hey! Quick update: good news - we got extra budget for the marketing campaign! 🎉"
    },
    {
        "input": "I am writing to inform you that the client presentation has been rescheduled to accommodate their availability.",
        "output": "Hey! Quick update: moved the client presentation to fit their schedule"
    },

    # Category 2: Meeting Requests (5 examples)
    {
        "input": "I would like to schedule a meeting to discuss the quarterly results and strategic planning for the next fiscal year. Please advise your availability.",
        "output": "Can we chat about Q4 results and next year's planning? What's your availability? 🗓️"
    },
    {
        "input": "I would like to schedule a brief call to review the current status of the implementation and address any concerns.",
        "output": "Can we hop on a quick call to review the implementation? Want to make sure we're aligned 📞"
    },
    {
        "input": "I would like to schedule a meeting with the development team to discuss the technical architecture for the new feature.",
        "output": "Can we get the dev team together to talk through the new feature's architecture?"
    },
    {
        "input": "I would like to schedule a one-on-one session to provide feedback on your recent work and discuss career development opportunities.",
        "output": "Can we schedule a 1:1? Want to chat about your recent work and career growth 🌱"
    },
    {
        "input": "I would like to schedule a stakeholder meeting to present the findings from the recent market research study.",
        "output": "Can we set up time with stakeholders to share the market research findings? 📈"
    },

    # Category 3: Issue Escalations (5 examples)
    {
        "input": "This matter requires immediate attention as it is impacting our ability to meet the project deadline. Please prioritize accordingly.",
        "output": "Heads up - we need to handle this ASAP, it's blocking us from hitting the deadline ⚠️"
    },
    {
        "input": "This matter requires immediate attention due to a critical security vulnerability that has been identified in the production environment.",
        "output": "Heads up - we've got a critical security issue in prod that needs immediate attention 🚨"
    },
    {
        "input": "This matter requires immediate attention as multiple clients have reported experiencing service disruptions.",
        "output": "Heads up - several clients are reporting service issues, we need to jump on this"
    },
    {
        "input": "This matter requires immediate attention because the vendor has informed us of a significant delay in component delivery.",
        "output": "Heads up - vendor just told us there's a big delay on components, need to figure out next steps"
    },
    {
        "input": "This matter requires immediate attention as we have discovered data inconsistencies that may affect reporting accuracy.",
        "output": "Heads up - found some data issues that could mess with our reports, let's fix this fast"
    },

    # Category 4: Follow-ups (5 examples)
    {
        "input": "Per our previous conversation regarding the budget allocation, I wanted to follow up on the status of the approval process.",
        "output": "Following up on our budget chat - any updates on the approval?"
    },
    {
        "input": "Per our previous conversation, I am following up to see if you had a chance to review the proposal I sent last week.",
        "output": "Following up on the proposal I sent last week - did you get a chance to look it over?"
    },
    {
        "input": "Per our previous conversation regarding the vendor selection, I wanted to confirm that we are proceeding with Option B as discussed.",
        "output": "Following up on vendor selection - confirming we're going with Option B, right?"
    },
    {
        "input": "Per our previous conversation, I wanted to check in on the timeline for completing the user acceptance testing.",
        "output": "Following up on UAT - what's the timeline looking like?"
    },
    {
        "input": "Per our previous conversation about the team restructure, I wanted to see if there are any updates you can share.",
        "output": "Following up on the team restructure discussion - any news you can share?"
    },

    # Category 5: Approvals (5 examples)
    {
        "input": "Please be advised that your request for additional resources has been approved by the management team and will be processed accordingly.",
        "output": "Just FYI - your request for extra resources got approved! Management team signed off ✅"
    },
    {
        "input": "Please be advised that the proposed changes to the project scope have been approved and we can proceed with implementation.",
        "output": "Just FYI - scope changes are approved, we're good to move forward 🚀"
    },
    {
        "input": "Please be advised that your vacation request has been approved for the dates specified.",
        "output": "Just FYI - your vacation is approved for those dates! 🏖️"
    },
    {
        "input": "Please be advised that the purchase order has been approved and the procurement process will commence shortly.",
        "output": "Just FYI - PO is approved, procurement will kick off soon"
    },
    {
        "input": "Please be advised that the new policy changes have been approved by leadership and will take effect next month.",
        "output": "Just FYI - leadership approved the new policy changes, goes live next month"
    },

    # Category 6: Delays (5 examples)
    {
        "input": "We regret to inform you that the product launch will be postponed until next quarter due to quality assurance concerns.",
        "output": "Small update - looks like we're pushing the product launch to next quarter due to some QA issues 😕"
    },
    {
        "input": "We regret to inform you that the interview process will take longer than anticipated due to scheduling conflicts with key stakeholders.",
        "output": "Small update - looks like the interview process will take a bit longer, scheduling has been tricky"
    },
    {
        "input": "We regret to inform you that the software deployment has been delayed due to compatibility issues discovered during final testing.",
        "output": "Small update - looks like we're delaying the deployment, found some compatibility issues in testing"
    },
    {
        "input": "We regret to inform you that the training session has been postponed to allow for additional preparation time.",
        "output": "Small update - looks like we're postponing the training to get better prepped"
    },
    {
        "input": "We regret to inform you that the contract signing has been delayed pending legal review.",
        "output": "Small update - looks like contract signing is on hold while legal reviews it"
    },

    # Category 7: Questions (5 examples)
    {
        "input": "I would appreciate your input on the best approach for handling this client relationship going forward.",
        "output": "Quick question - what do you think is the best way to handle this client relationship?"
    },
    {
        "input": "I would appreciate your input on whether we should prioritize feature development or technical debt reduction in the next sprint.",
        "output": "Quick question - should we focus on new features or tech debt next sprint? What's your take?"
    },
    {
        "input": "I would appreciate your input on the most effective marketing channels for reaching our target demographic.",
        "output": "Quick question - which marketing channels do you think would work best for our target audience?"
    },
    {
        "input": "I would appreciate your input on whether we need to hire additional team members or redistribute current workload.",
        "output": "Quick question - do you think we need to hire more people or just shuffle the current workload?"
    },
    {
        "input": "I would appreciate your input on the pros and cons of migrating to the cloud infrastructure at this time.",
        "output": "Quick question - thoughts on cloud migration right now? Pros/cons?"
    },

    # Category 8: Feedback (5 examples)
    {
        "input": "I would like to provide feedback on your recent presentation. The content was comprehensive, though the delivery could benefit from more audience engagement.",
        "output": "Thoughts on your presentation - content was solid! For next time, maybe add more audience interaction?"
    },
    {
        "input": "I would like to provide feedback on the current workflow process. It appears there are several bottlenecks that could be optimized.",
        "output": "Thoughts on our workflow - seems like there are a few bottlenecks we could smooth out"
    },
    {
        "input": "I would like to provide feedback on the draft proposal. The structure is strong but some sections need additional supporting data.",
        "output": "Thoughts on the proposal draft - structure looks good! Just need more data in a few sections"
    },
    {
        "input": "I would like to provide feedback on the team's recent sprint. The velocity was impressive, though code quality standards need more attention.",
        "output": "Thoughts on the last sprint - great velocity! Let's make sure we keep code quality high too 💪"
    },
    {
        "input": "I would like to provide feedback on the customer support response times. We are seeing improvements but there is still room for enhancement.",
        "output": "Thoughts on support response times - we're getting better! Still some room to improve though"
    },

    # Category 9: Clarifications (5 examples)
    {
        "input": "For the sake of clarity, I wanted to confirm that the deadline for submission is end of business on Friday, not Monday as previously discussed.",
        "output": "Just to clarify - deadline is EOD Friday, not Monday like we talked about before"
    },
    {
        "input": "For the sake of clarity, the budget allocation covers operational expenses only and does not include capital expenditures.",
        "output": "Just to clarify - the budget is for operational stuff only, not capex"
    },
    {
        "input": "For the sake of clarity, all team members are expected to attend the mandatory training session, not just new hires.",
        "output": "Just to clarify - the training session is for everyone, not just new folks"
    },
    {
        "input": "For the sake of clarity, the new policy applies to all contractors as well as full-time employees.",
        "output": "Just to clarify - new policy covers contractors too, not just FTEs"
    },
    {
        "input": "For the sake of clarity, the performance metrics discussed are for the entire quarter, not just the monthly figures.",
        "output": "Just to clarify - those metrics are for the whole quarter, not just this month"
    },

    # Category 10: Acknowledgments (5 examples)
    {
        "input": "We acknowledge receipt of your request and will process it within the standard timeframe of three to five business days.",
        "output": "Got it! We'll process your request within 3-5 business days ✅"
    },
    {
        "input": "We acknowledge receipt of your feedback and will incorporate the suggestions into the next iteration of the product.",
        "output": "Got it! We'll work your feedback into the next version 👍"
    },
    {
        "input": "We acknowledge receipt of your concern regarding the billing discrepancy and are investigating the matter immediately.",
        "output": "Got it! Looking into the billing issue right now"
    },
    {
        "input": "We acknowledge receipt of your application and will be in touch within two weeks regarding next steps.",
        "output": "Got it! We'll reach out within 2 weeks about next steps"
    },
    {
        "input": "We acknowledge receipt of the signed contract and will proceed with project initiation as outlined in the agreement.",
        "output": "Got it! Contract received, we'll kick off the project as planned 🚀"
    }
]


def generate_training_data() -> List[Dict[str, str]]:
    """Generate training data in instruction-input-output format."""

    instruction = "Rewrite this corporate email into a casual, friendly Slack message."

    training_data = []
    for template in TEMPLATES:
        training_data.append({
            "instruction": instruction,
            "input": template["input"],
            "output": template["output"]
        })

    return training_data


def save_jsonl(data: List[Dict], output_path: str) -> None:
    """Save data to JSONL format."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    print(f"✅ Saved {len(data)} examples to {output_path}")


def validate_data(data: List[Dict]) -> None:
    """Validate training data quality."""

    print("\n📊 Data Validation:")
    print(f"  Total examples: {len(data)}")

    # Check for required fields
    for i, item in enumerate(data):
        assert "instruction" in item, f"Example {i} missing 'instruction'"
        assert "input" in item, f"Example {i} missing 'input'"
        assert "output" in item, f"Example {i} missing 'output'"

    print("  ✅ All examples have required fields")

    # Check input lengths
    input_lengths = [len(item["input"].split()) for item in data]
    avg_input_length = sum(input_lengths) / len(input_lengths)
    print(f"  Average input length: {avg_input_length:.1f} words")

    # Check output is shorter than input
    shorter_outputs = sum(1 for item in data if len(item["output"]) < len(item["input"]))
    print(f"  Outputs shorter than inputs: {shorter_outputs}/{len(data)}")

    # Check for emojis
    emoji_count = sum(1 for item in data if any(char in item["output"] for char in "😅🎉✅📊📞🗓️⚠️🚨🏖️😕💪👍🚀"))
    print(f"  Examples with emojis: {emoji_count}/{len(data)} ({emoji_count/len(data)*100:.1f}%)")

    # Check for duplicates
    inputs = [item["input"] for item in data]
    unique_inputs = len(set(inputs))
    print(f"  Unique inputs: {unique_inputs}/{len(data)}")

    assert unique_inputs == len(data), "Found duplicate inputs!"
    print("  ✅ No duplicates found")

    print("\n✅ Validation passed!\n")


def main():
    """Main execution function."""

    print("🚀 Generating training data...\n")

    # Generate data
    training_data = generate_training_data()

    # Validate
    validate_data(training_data)

    # Save
    output_path = "data/train.jsonl"
    save_jsonl(training_data, output_path)

    # Show sample
    print("\n📝 Sample example:")
    sample = training_data[0]
    print(f"  Input:  {sample['input'][:80]}...")
    print(f"  Output: {sample['output']}")
    print(f"\n✅ Data generation complete!")


if __name__ == "__main__":
    main()
