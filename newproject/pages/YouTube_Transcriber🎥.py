import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Load environment variables
load_dotenv()

# Configure the Google Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the prompt for generating summary
prompt = """You are a YouTube video summarizer. You will be taking the transcript text 
and summarizing the entire video and providing the important summary in points 
within 250 words. Please provide the summary of the text given here: """

# Function to get the transcript details from YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        # Extract the video ID from the YouTube URL using regex
        video_id = re.search(r"(?<=v=)[a-zA-Z0-9_-]+", youtube_video_url)
        if video_id:
            video_id = video_id.group(0)
        else:
            st.error("🚨 Invalid YouTube URL! Please check and try again.")
            return None
        
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all transcript text into a single string
        transcript = " ".join([i["text"] for i in transcript_text])
        
        return transcript, video_id
    except Exception as e:
        st.error(f"⚠️ Error fetching transcript: {e}")
        return None, None

# Function to generate content using Google Gemini Pro
def generate_gemini_content(transcript_text, prompt, length):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + transcript_text)
        summary = response.text

        if length == "Short":
            summary = summary[:150]  # Cut down summary to 150 words
        elif length == "Long":
            summary += " [This is a long version of the summary.]"

        return summary
    except Exception as e:
        st.error(f"❌ Error generating summary: {e}")
        return None

# Streamlit UI Setup
st.set_page_config(page_title="YouTube Transcript to Summary Converter", layout="wide")
st.title("🎥 YouTube Transcript to Summary Converter ✨")
st.markdown("Enter a YouTube video link to convert the transcript to a detailed summary!")

# User input for YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link:", placeholder="e.g. https://www.youtube.com/watch?v=example", help="Paste a YouTube link here to fetch the transcript.")

# Show thumbnail image from the video if link is entered
if youtube_link:
    video_id = youtube_link.split("=")[-1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", caption="Video Thumbnail", width=250)  # Smaller size

# Sidebar for FAQ Section
st.sidebar.header("🛠️ FAQ")
st.sidebar.markdown("""
- **How do I use this tool?**  
  Enter a YouTube video link, and the app will fetch the transcript and provide a summary.
  
- **Why is the summary not accurate?**  
  The summary depends on the transcript quality. If it's incomplete, the summary might not be perfect.

- **Can I download the summary?**  
  Yes, you can download the summary as a text file.
""")

# Dropdown to select summary length
summary_length = st.selectbox("📝 Select Summary Length", ["Short", "Medium", "Long"], index=1, help="Choose the length of your summary.")

# Button to get detailed notes
if st.button("🔍 Get Detailed Notes"):
    if youtube_link:
        with st.spinner("⏳ Fetching transcript and generating summary..."):
            transcript_text, video_id = extract_transcript_details(youtube_link)
            
            if transcript_text:
                summary = generate_gemini_content(transcript_text, prompt, summary_length)
                
                if summary:
                    st.markdown("## 📑 Detailed Notes:")
                    st.write(summary)

                    # Word count of the summary
                    st.markdown(f"### 📝 Word Count: {len(summary.split())} words")

                    # Option to download the summary
                    st.download_button(
                        label="💾 Download Summary as Text",
                        data=summary,
                        file_name="video_summary.txt",
                        mime="text/plain"
                    )
            else:
                st.warning("⚠️ Transcript could not be fetched. Please check the video URL.")
    else:
        st.warning("🚨 Please enter a valid YouTube video URL.")

# Clear button to reset inputs and output
if st.button("🧹 Clear All"):
    st.experimental_rerun()

# Custom CSS Styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #007BFF;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stTextInput>div>input {
            font-size: 16px;
        }
        .stImage>img {
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# Explanation of AI usage
st.markdown("""
    ## How it Works:
    - The summary length can be customized based on user preference.
""")
