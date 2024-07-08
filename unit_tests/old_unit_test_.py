
from deepeval.test_case import LLMTestCase
import deepeval
from deepeval import assert_test
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from generate_source import SemanticSearchService
import pytest



weaviate_url = "http://localhost:8777"
service = SemanticSearchService(weaviate_url)
Responses = [
    {
        "Object": 1,
        "ID": "00a4917b-e25f-43da-9771-bc4cf742c3ec",
        "Question": "where is the help desk for tech issues",
        "Answer": "The Help Desk for tech issues is located above the fitness center. You can also create a Help Desk ticket using their portal or by emailing support@hhmi.org.",
        "Response Notes": "Check for email support@hhmi.org",
        "Source Notes": "Wiki: Getting Help (https://wikis.janelia.org/display/SCSW/Getting+Help)",
        "Context": "For assistance with computer, network, or phone issues, you can create a Help Desk ticket either by using their portal or by emailing support@hhmi.org. Additionally, you can visit the Help Desk in person, which is located above the fitness center. To get there, walk out the main doors and go across to the hotel."
    },
    {
        "Object": 4,
        "ID": "024cf273-1920-43cb-a7e8-c3191af81c59",
        "Question": "Where do i go if i need tech help at janelia",
        "Answer": "If you need tech help at Janelia, you should create a Help Desk ticket either by using their portal or by emailing support@hhmi.org. You can also walk over to the Help Desk, which is located above the fitness center.",
        "Response Notes": "Look for help desk and support@hhmi.org",
        "Source Notes": "Wiki: Getting Help (https://wikis.janelia.org/display/SCSW/Getting+Help)",
        "Context": "For assistance with computer, network, or phone issues, you can create a Help Desk ticket either by using their portal or by emailing support@hhmi.org. Additionally, you can visit the Help Desk in person, which is located above the fitness center. To get there, walk out the main doors and go across to the hotel."
    },
    {
        "Object": 6,
        "ID": "02e95fd0-fbfc-461b-bec9-7bb1278ba7bb",
        "Question": "What is mdas",
        "Answer": "MDAS stands for \"microscope direct attached storage.\" It is a specialized storage system designed to handle high-speed data capture from microscopes, providing a distributed and shared storage solution. The system includes a tiered storage approach with SSDs for incoming writes and larger spinning disk pools for long-term storage.",
        "Response Notes": "None",
        "Source Notes": "Wiki: Check if a Source points to a 11:44 slack message in #storage",
        "Context": "MDAS stands for 'microscope direct attached storage.' It is a storage system designed to handle high-speed data capture directly from microscopes. The system includes a tiered storage approach, with an SSD tier for incoming writes to avoid bottlenecks and a larger spinning disk pool for longer-term storage. This setup allows for efficient data management and transfer, although it may require expansion to sustain higher data rates and concurrent access."
    },
    {
        "Object": 13,
        "ID": "0d0fad42-ba6a-4223-8bd2-2e58554d51cb",
        "Question": "what slack channels should i join if i'm new",
        "Answer": "You might want to join channels like #slack-best-practices for questions, answers, and useful tips, as well as other interesting channels that you can find by clicking the '+' next to \"Channels\" and choosing \"Browse channels.\"",        
        "Response Notes": "Completely misleading response and we dont know why. Should mention python according to sources",
        "Source Notes": "NA",
        "Context": "If you're new at Janelia, you might want to explore and join various Slack channels to get integrated into the community. You can click the '+' next to 'Channels' and choose 'Browse channels' to see all available options. Additionally, channels related to Janelia-Software and other public channels within the HHMI workspace are accessible to anyone at Janelia. If you encounter any issues joining a channel, you can contact Donald Olbris for assistance."
    },
    {
        "Object": 25,
        "ID": "189988ac-433e-4640-9684-5c46f40e54b5",
        "Question": "Who knows the most about julia?",
        "Answer": "Mark Kittisopikul appears to know the most about Julia, as he has shared multiple resources and information about the programming language.",
        "Response Notes": "Only mentions one of two strong julia programmers",
        "Source Notes": "NA",
        "Context": "Mark Kittisopikul and Ben Arthur appear to have significant knowledge about Julia at Janelia. Mark has shared resources and insights about Julia, while Ben has historically used it for scientific computing."
    },
    {
        "Object": 41,
        "ID": "2fa07b48-9130-458d-a38f-ae4860a901e3",
        "Question": "how do I get help from cryoem?",
        "Answer": "To get help from the CryoEM facility, you need to contact the facility to initiate the registration process. Once registered, you can use their online scheduler to book equipment time if you have ready-to-image cryo grids. If you need more assistance, have technical questions, or want to initiate a collaboration, you can contact the facility director, Dr. Zhiheng Yu, at yuz10@janelia.hhmi.org, or email cryoemfacility@janelia.hhmi.org.",
        "Response Notes": "Includes yuz10@janelia.hhmi.org or cryoemfacility@janelia.hhmi.org",
        "Source Notes": "first source should be \"Web: Cryo-Electron Microscopy\"",
        "Context": "To get help from the CryoEM facility at Janelia, you need to follow a few steps. First, if you are a new user, you should contact the facility to initiate the registration process. Once registered, you can use the online scheduler to book equipment time if you have ready-to-image cryo grids. If you need more assistance, have technical questions, or want to initiate a collaboration, you can contact the facility director, Dr. Zhiheng Yu, or email the facility directly. The facility offers a range of services, including high-quality data collection, cryo FIB milling, and collaboration on various aspects of the cryoEM workflow. Make sure to review the policies and instructions before reserving equipment time or sending any shipments."
    },
    {
        "Object": 49,
        "ID": "38af623d-35d3-48ab-8587-a40aa8d2694b",
        "Question": "who are the group leaders in 4DCP?",
        "Answer": "The group leaders in 4D Cellular Physiology (4DCP) are outstanding scientists who lead laboratories developing new tools or methods or bringing innovative approaches to the study of biological problems within Janelia's research areas. They are passionate about conducting research in various scientific areas, including cell biological mechanisms of development, cytoskeleton and signaling processes in tissues, deep tissue and intravital imaging, and more. These leaders are typically late-stage graduate students, postdoctoral associates, or early-career faculty members enthusiastic about taking charge of a small lab in a collaborative environment.",
        "Response Notes": "Does not return any specific people",
        "Source Notes": "Will not work until we segment over the 4DCP website in the future",
        "Context": "The group leaders in 4D Cellular Physiology (4DCP) are: JENNIFER LIPPINCOTT-SCHWARTZ, Head of 4D Cellular Physiology; MISHA AHRENS, Senior Group Leader; MENG WANG, Senior Group Leader; ISABEL ESPINOSA MEDINA, Group Leader; ANOJ ILANGES, Group Leader; JIEFU LI, Group Leader; YIN LIU, Group Leader; ALLYSON SGRO, Group Leader; ALLYSON SGRO, Group Leader."    
    },
    {
        "Object": 50,
        "ID": "38af623d-35d3-48ab-8587-a40aa8d2694b",
        "Question": "what is neuronbridge",
        "Answer": "NeuronBridge is a web application designed to facilitate the search for morphological matches between large data sets of neurons imaged using different modalities. It is particularly useful in neuroscience research involving Drosophila, where it helps researchers find similar neurons across various imaging techniques, such as electron microscopy and confocal light microscopy. The application features a user-friendly graphical interface, an extensible data model, a serverless cloud architecture, and a highly parallel image search engine. It supports the integration and reuse of its software code, open APIs, and processed data sets, and is available as a service to the neuroscience research community.",
        "Response Notes": "This current response is great, use it as baseline",
        "Source Notes": "Main source isn't very helpful (Wiki: NeuronBridge) only had links, future if we sort by content of the website and link name",
        "Context": "NeuronBridge is a project developed at Janelia that provides a platform for managing and searching neuron imagery. It includes a front-end website and back-end AWS services, and it supports color depth searches. The platform is designed to handle various stages of development, including production, internal pre-release, and validation. It also utilizes multiple AWS S3 buckets for storing metadata, precomputed search JSON files, and web content. The project is actively maintained with source control repositories available on GitHub for different components of the system."
    },
    {
        "Object": 54,
        "ID": "3afec7cb-9a93-478f-ae0b-fa1815f57395",
        "Question": "how many people are there in Misha Arhens lab?",
        "Answer": "There are 18 people in Misha Ahrens' lab.",
        "Response Notes": "There are not 18 people in the lab. Check with the data in ./data. Also, paste website content into gpt and ask the same question",
        "Source Notes": "N/A",
        "Context": "There are 7 members in the Misha Arhens lab team, 2 visiting scientists in residence,  11 other science contributors, and 1 lab coordinator"
    }
]

@pytest.fixture
def setup_test_cases():
    # Setup code to create test cases
    test_cases = create_test_cases(Responses)
    return test_cases

@pytest.fixture
def setup_metric():
    # Setup code for metric
    metric = HallucinationMetric(threshold=0.95, model="gpt-3.5-turbo")
    return metric

def create_test_cases(Responses):
    test_cases = []
    for response in Responses:
        input_question = response['Question']
        #             = service.generate_response(input_question)  # Assuming generate_response can take context
        actual_output = response['Answer']
        context = [response['Context']]  # Extract context and make it a list
        print (context)

        test_case = LLMTestCase(
            input=input_question,
            actual_output=actual_output,
            context=[],  # Pass context here
        )
        test_cases.append(test_case)
    return test_cases

def test_generate_response(setup_test_cases, setup_metric):
    results = []
    for test_case in setup_test_cases:
        # Assuming the correct method name is 'assess'
        metric_result = setup_metric.assess(test_case)
        important_features_present = check_important_features(test_case['response'])
        
        assert metric_result.passed, "Test case failed the hallucination metric"
        assert important_features_present, "Important features are missing in the response"
        
        results.append({
            'test_case': test_case,
            'passed_hallucination_metric': metric_result.passed,
            'important_features_present': important_features_present,
        })
    
    print(results)


def check_important_features(response):
    # Implement your logic to check for important features in the response
    # For example, checking if certain keywords exist:
    important_keywords = ['keyword1', 'keyword2']
    return any(keyword in response for keyword in important_keywords)

# This does not work because the inputs to assert_test seem to be wrong
# def test_generate_response():
#     test_cases = create_test_cases(Responses)
#     for test_case in test_cases:
#         metric = HallucinationMetric(threshold=0.95, model="gpt-3.5-turbo")
#         assert_test(
#             test_case=test_case,  # Pass the single test case directly
#             metric=metric,  # Pass the single metric directly
#             hyperparameters={"model": "gpt-3.5-turbo", "prompt template": "You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines."},
#         )
