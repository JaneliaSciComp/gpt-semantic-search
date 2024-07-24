import streamlit as st
import pandas as pd
import sys
import sys
from pathlib import Path

target_dir_path = Path(__file__).resolve().parent / '../unit_tests/'
sys.path.append(str(target_dir_path))
from unit_tests.test_cases import custom_checking, Responses, service_ans, service_src

st.set_page_config(page_title="Custom Test Runner", layout="wide", page_icon="🧪")

st.title("Custom Test Runner")

if st.button("Run Tests"):
    test_cases, data_list = custom_checking(Responses)

    st.header("Test Results")

    results_df = pd.DataFrame(columns=["Test ID", "Question", "Output", "Sources", "Result", "Missing"])
    rows_to_add = []

    for data in data_list:
        question, output, sources, test_id, missing = data
        result = "PASSED" if test_cases[test_id] else "FAILED"
        missing = ', '.join(data[4]) if len(data) > 4 and not test_cases[test_id] else ''
        rows_to_add.append({
            "Test ID": test_id,
            "Question": question,
            "Output": output[:100] + "...",  # Truncate for readability
            "Sources": sources[:100] + "...",  # Truncate for readability
            "Result": result,
            "Missing": ', '.join(data[4]) if len(data) > 4 and not test_cases[test_id] else ''
        })
    new_rows_df = pd.DataFrame(rows_to_add)
    results_df = pd.concat([results_df, new_rows_df], ignore_index=True)
    
    st.dataframe(results_df.style.applymap(
        lambda x: 'background-color: #90EE90' if x == 'PASSED' else 'background-color: #FFB3BA',
        subset=['Result']
    ))

    passed = sum(test_cases.values())
    total = len(test_cases)
    st.subheader(f"Summary: Passed {passed}/{total} ({passed/total*100:.2f}%)")

    st.header("Detailed Results")
    for test_id, result in test_cases.items():
        with st.expander(f"Test ID: {test_id} - {'PASSED' if result else 'FAILED'}"):
            data = next(data for data in data_list if data[-2] == test_id)
            st.write(f"Question: {data[0]}")
            st.write(f"Output: {data[1]}")
            st.write(f"Sources: {data[2]}")
            
            if not data[4]:  # Check if the list is empty
                st.write("Nothing is missing, passed!")
            else:
                st.write(f"Missing: {', '.join(data[4])}")

st.sidebar.info("Click 'Run Tests' to execute the custom tests and view the results.")
