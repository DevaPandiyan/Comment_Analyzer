# Use a pipeline as a high-level helper
from transformers import pipeline
import streamlit as st

if 'all_comments' not in st.session_state:
    st.session_state['all_comments'] = []

model_path = ("Models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots"
                  "/714eb0fa89d2f80546fda750413ed43d93601a13")
classifier = pipeline("text-classification", model=model_path)

st.title("Comments Analyzer")
st.video("https://www.youtube.com/watch?v=4_9JLsA63IA")
comment = st.text_input("Enter your Comment", value="amazing Video")


def classifier_comments(classifier, comment):
    result = classifier(comment)[0]['label']
    return result


def display_comments(comments):
    for idx, comment in enumerate(comments):
        # Determine background color based on sentiment
        if comment['classification'] == 'POSITIVE':
            background_color = '#d0f0c0'  # light green
        elif comment['classification'] == 'NEGATIVE':
            background_color = '#ffb6b6'  # light red
        else:
            background_color = '#ffffff'  # default white

        # Define the CSS styles for the comment box
        style = f"""
               background-color: {background_color};
               padding: 5px;
               color:black;
               border-radius: 10px;
               box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
               margin-bottom: 14px;
           """

        comment_html = f"""
               <div style="{style}">
                   <p style="margin:0"><strong> Deva </strong></p>
                   <p>{comment['comment']}</p>
               </div>
               """
        # Write the HTML code for the comment box
        st.write(comment_html, unsafe_allow_html=True)


# Create a selectbox to filter comments
filter_option = st.selectbox("Filter Comments", ["All", "Positive", "Negative", "None"])

# Filter comments based on selected option
filtered_comments = st.session_state['all_comments']
if filter_option == "Positive":
    filtered_comments = [c for c in filtered_comments if c['classification'] == 'POSITIVE']
elif filter_option == "Negative":
    filtered_comments = [c for c in filtered_comments if c['classification'] == 'NEGATIVE']
elif filter_option == "None":
    filtered_comments = [c for c in filtered_comments if c['classification'] == 'NONE']

if st.button("Add Comment"):
    comment_dic = {'comment': comment, 'classification': classifier_comments(classifier, comment)}
    st.session_state['all_comments'].append(comment_dic)



# Display filtered comments
display_comments(filtered_comments)
