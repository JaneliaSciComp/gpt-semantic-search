import weaviate
import streamlit as st
st.set_page_config(page_title="JaneliaGPT - Survey Responses", page_icon="📋")

from state import init_state
init_state()

# Initialize the client
client = weaviate.Client("http://search.int.janelia.org:8777")

# Initialize variables for pagination


# all_objects now contains all objects from Weaviate


# Fetch data and extract the list of objects
# Assuming 'client' is already initialized and available

def fetch_all_survey_responses(client, class_name="SurveyResponses", limit=100):
    all_responses = []
    offset = 0

    while True:
        # Fetch a batch of responses
        response = client.data_object.get(class_name=class_name, limit=limit, offset=offset)
        batch_responses = response["objects"]  # Adjust based on actual structure

        # Add the fetched responses to the list
        all_responses.extend(batch_responses)

        # Check if we've fetched all responses
        if len(batch_responses) < limit:
            break  # Exit loop if fewer responses than the limit are returned

        # Prepare for the next iteration
        offset += limit

    return all_responses

# Use the function to fetch all responses
all_objects = fetch_all_survey_responses(client)

# Example of rendering each response (adjust based on your needs)
  # Replace this with your rendering logic
def display_objects(objects):
    if (st.session_state["admin_toggle"]==True):
        for index, obj in enumerate(objects, start=1):
            st.markdown(f"### Object Number: {index}")        
            st.markdown(f"###### Class: {obj['class']}")
            st.markdown("##### Details", unsafe_allow_html=True)
            st.markdown(f"<small>ID: {obj['id']}</small>", unsafe_allow_html=True)
            st.markdown(f"<small>Creation Time: {obj['creationTimeUnix']}</small>", unsafe_allow_html=True)
            st.markdown(f"<small>Last Update Time: {obj['lastUpdateTimeUnix']}</small>", unsafe_allow_html=True)

            st.markdown("##### Properties", unsafe_allow_html=True)
            st.markdown(f"<h3 style='margin-bottom:0;'>Query:</h3><p style='font-size:20px; color:#FFC0CB;'>{obj['properties']['query']}</p>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='margin-bottom:0;'>Response:</h3><p style='font-size:20px; color:#FFC0CB;'>{obj['properties']['response']}</p>", unsafe_allow_html=True)

            # Dropdown for Survey
            survey_options = ["Unknown", "True", "False"]
            current_survey_value = obj['properties']['survey']
            if current_survey_value not in survey_options:
                current_survey_value = "Unknown"  # Fallback in case of unexpected value
            selected_survey_value = st.selectbox("Survey", options=survey_options, index=survey_options.index(current_survey_value), key=f"survey_{index}")

            # Check if the selected value is different from the current value
            if selected_survey_value != current_survey_value:
                # Update the object in the database
                updated_properties = obj['properties']
                updated_properties['survey'] = selected_survey_value
                client.data_object.update(
                    data_object=updated_properties,
                    class_name="SurveyResponses",
                    uuid=obj['id']
                )
                st.success("Survey value updated successfully!")

            # Add space after each object
            st.markdown("---")  # Horizontal line as a separator
    else:
        st.error("You do not have permission to view this page.")
# Assuming all_objects is defined and contains your objects

display_objects(all_objects)

# 