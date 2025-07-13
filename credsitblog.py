import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import zipfile
import base64
from io import BytesIO
from docx import Document
from huggingface_hub import InferenceClient
import os
from urllib.parse import urlparse

# Configure page
st.set_page_config(page_title="Enhanced SEO Content Automation", page_icon="📚", layout="wide")

# Initialize session state
if "articles" not in st.session_state:
    st.session_state["articles"] = {}
if "uploaded_articles" not in st.session_state:
    st.session_state["uploaded_articles"] = {}
if "article_metadata" not in st.session_state:
    st.session_state["article_metadata"] = {}
if "images" not in st.session_state:
    st.session_state["images"] = {}
if "publish_log" not in st.session_state:
    st.session_state["publish_log"] = []
if "existing_posts" not in st.session_state:
    st.session_state["existing_posts"] = []
if "custom_wp_config" not in st.session_state:
    st.session_state["custom_wp_config"] = {}

def init_hf_client():
    """Initialize Hugging Face client"""
    try:
        HF_TOKEN = st.secrets.get("HF_TOKEN")
        if not HF_TOKEN:
            return None
        return InferenceClient(
            model="stabilityai/stable-diffusion-3-medium",
            token=HF_TOKEN
        )
    except Exception as e:
        st.error(f"Error initializing HF client: {str(e)}")
        return None

def get_api_key(provider):
    """Get API key for selected provider"""
    if provider == "Grok (X.AI)":
        return st.secrets.get("GROK_API_KEY")
    elif provider == "OpenAI":
        return st.secrets.get("OPENAI_API_KEY")
    return None

def extract_title_from_content(content):
    """Extract title from article content"""
    # Try to find H1 tag first
    h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', content, re.IGNORECASE | re.DOTALL)
    if h1_match:
        return re.sub(r'<[^>]+>', '', h1_match.group(1)).strip()
    
    # Try to find title tag
    title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
    if title_match:
        return re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
    
    # Get first line as title (remove HTML tags)
    first_line = content.split('\n')[0]
    return re.sub(r'<[^>]+>', '', first_line).strip()[:100]

def extract_keywords_from_content(content):
    """Extract potential keywords from content"""
    # Remove HTML tags
    clean_content = re.sub(r'<[^>]+>', '', content)
    
    # Find words that appear frequently (simple keyword extraction)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', clean_content.lower())
    word_freq = {}
    for word in words:
        if word not in ['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'said', 'each', 'which', 'their', 'time', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'only', 'come', 'year', 'work', 'such', 'make', 'even', 'most', 'after', 'good', 'other', 'many', 'well', 'some', 'could', 'would', 'also', 'back', 'there', 'through', 'where', 'much', 'about', 'before', 'right', 'being', 'should', 'people', 'these', 'article', 'content']:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Return top 5 keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:5] if freq > 2]

def generate_metadata_for_article(content, title, api_key, provider):
    """Generate SEO metadata for uploaded article using AI"""
    keywords = ', '.join(extract_keywords_from_content(content))
    
    prompt = f"""
Analyze this article content and generate SEO metadata:

Title: {title}
Content Preview: {content[:1000]}...
Detected Keywords: {keywords}

Generate:
1. SEO-optimized title (60 chars max, include year 2024/2025)
2. Meta description (150-160 chars)
3. Primary keyword (1-3 words)
4. Secondary keywords (5 keywords)
5. Content category (Guide/Tutorial/Review/Comparison/How-to/FAQ)
6. Search volume estimate (High/Medium/Low) for Indian market
7. Tags for WordPress (5-8 tags)

Respond in JSON format:
{{
  "seo_title": "Optimized SEO Title Here",
  "meta_description": "Compelling meta description...",
  "primary_keyword": "main keyword",
  "secondary_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "category": "Guide",
  "search_volume": "Medium",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}
"""
    
    # Configure API based on provider
    if provider == "Grok (X.AI)":
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        models_to_try = ["grok-3-latest", "grok-2-1212", "grok-2-latest", "grok-beta", "grok-2"]
        
        for model_name in models_to_try:
            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            
            try:
                response = requests.post(url, json=body, headers=headers)
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    try:
                        return json.loads(content)
                    except:
                        # Fallback metadata
                        return {
                            "seo_title": title,
                            "meta_description": f"Complete guide about {title}. Learn everything you need to know.",
                            "primary_keyword": keywords.split(',')[0] if keywords else title.split()[0],
                            "secondary_keywords": keywords.split(',')[:5] if keywords else [],
                            "category": "Guide",
                            "search_volume": "Medium",
                            "tags": keywords.split(',')[:5] if keywords else []
                        }
                elif response.status_code == 404:
                    continue
                else:
                    continue
            except Exception as e:
                continue
        
        return None
        
    elif provider == "OpenAI":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, json=body, headers=headers)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except:
                    return {
                        "seo_title": title,
                        "meta_description": f"Complete guide about {title}. Learn everything you need to know.",
                        "primary_keyword": keywords.split(',')[0] if keywords else title.split()[0],
                        "secondary_keywords": keywords.split(',')[:5] if keywords else [],
                        "category": "Guide",
                        "search_volume": "Medium",
                        "tags": keywords.split(',')[:5] if keywords else []
                    }
            else:
                return None
        except Exception as e:
            return None
    
    return None

def generate_optimized_image_prompt(title, content, primary_keyword, api_key, provider):
    """Generate optimized image prompt using AI"""
    prompt = f"""
Create an optimized image generation prompt for:
Title: {title}
Primary Keyword: {primary_keyword}
Content Type: Blog article/educational content

Consider Hugging Face Stable Diffusion limitations:
- Simple, clear descriptions work best
- Avoid complex scenes
- Focus on single main subject
- Use professional, clean style keywords

Generate a concise, effective prompt (max 50 words) that will create a high-quality, professional image suitable for a blog article header.

Examples of good prompts:
- "Professional infographic design, clean layout, modern business theme, blue and white colors, minimalist style"
- "Educational illustration, simple diagram style, professional design, clean background"
- "Modern website mockup, clean interface design, professional layout, blue gradient background"

Create similar prompt for the given topic. Return only the prompt text, no quotes or explanations.
"""
    
    # Configure API based on provider
    if provider == "Grok (X.AI)":
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        models_to_try = ["grok-3-latest", "grok-2-1212", "grok-2-latest", "grok-beta", "grok-2"]
        
        for model_name in models_to_try:
            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            try:
                response = requests.post(url, json=body, headers=headers)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip().replace('"', '')
                elif response.status_code == 404:
                    continue
            except Exception as e:
                continue
        
        # Fallback prompt
        return f"Professional illustration about {primary_keyword}, clean design, modern style, educational content, high quality"
        
    elif provider == "OpenAI":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        try:
            response = requests.post(url, json=body, headers=headers)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip().replace('"', '')
            else:
                return f"Professional illustration about {primary_keyword}, clean design, modern style, educational content, high quality"
        except Exception as e:
            return f"Professional illustration about {primary_keyword}, clean design, modern style, educational content, high quality"
    
    return f"Professional illustration about {primary_keyword}, clean design, modern style, educational content, high quality"

def generate_ai_image(prompt, hf_client):
    """Generate image using Hugging Face Inference Client"""
    if not hf_client:
        st.error("Hugging Face client not initialized")
        return None
    
    try:
        image = hf_client.text_to_image(prompt)
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return img_buffer
    except Exception as e:
        st.error(f"Image generation error: {str(e)}")
        return None

def get_wordpress_posts(wp_config, per_page=100):
    """Fetch existing WordPress posts"""
    try:
        auth_str = f"{wp_config['username']}:{wp_config['password']}"
        auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
        headers = {"Authorization": f"Basic {auth_token}"}
        
        url = f"{wp_config['base_url']}/wp-json/wp/v2/posts?per_page={per_page}&status=publish"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            posts = response.json()
            return [{"id": post["id"], "title": post["title"]["rendered"], "link": post["link"], "content": post["content"]["rendered"][:500]} for post in posts]
        else:
            st.error(f"Failed to fetch posts: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching posts: {str(e)}")
        return []

def find_internal_linking_opportunities(article_content, existing_posts, api_key, provider):
    """Use AI to find internal linking opportunities"""
    post_list = "\n".join([f"- {post['title']}: {post['link']}" for post in existing_posts[:20]])
    
    prompt = f"""
Analyze this article content and find internal linking opportunities from existing posts:

Article Content Preview: {article_content[:1000]}...

Existing Posts:
{post_list}

Find 3-5 relevant posts that should be linked from this article. Consider:
1. Topic relevance and context
2. Natural linking opportunities
3. User experience and value

Return JSON format:
{{
  "links": [
    {{
      "anchor_text": "relevant phrase from article",
      "target_post": "Post Title",
      "target_url": "https://...",
      "reason": "Why this link is relevant"
    }}
  ]
}}

Only suggest high-quality, relevant links. Maximum 5 links.
"""
    
    # Use same API logic as other functions
    if provider == "Grok (X.AI)":
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        models_to_try = ["grok-3-latest", "grok-2-1212", "grok-2-latest", "grok-beta", "grok-2"]
        
        for model_name in models_to_try:
            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            
            try:
                response = requests.post(url, json=body, headers=headers)
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    try:
                        return json.loads(content)
                    except:
                        return {"links": []}
                elif response.status_code == 404:
                    continue
            except Exception as e:
                continue
        
        return {"links": []}
        
    elif provider == "OpenAI":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, json=body, headers=headers)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except:
                    return {"links": []}
        except Exception as e:
            return {"links": []}
    
    return {"links": []}

def apply_internal_links_to_content(content, links_data):
    """Apply internal links to content"""
    modified_content = content
    
    for link in links_data.get("links", []):
        anchor_text = link["anchor_text"]
        target_url = link["target_url"]
        
        # Replace first occurrence (case-insensitive)
        pattern = re.compile(rf"\b({re.escape(anchor_text)})\b", re.IGNORECASE)
        modified_content, n = pattern.subn(
            rf'<a href="{target_url}" target="_blank">\1</a>',
            modified_content,
            count=1
        )
    
    return modified_content

def publish_to_wordpress(title, content, image_buffer, tags, wp_config, publish_now=True):
    """Publish article to WordPress"""
    auth_str = f"{wp_config['username']}:{wp_config['password']}"
    auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
    headers = {"Authorization": f"Basic {auth_token}"}
    
    img_id = None
    
    # Upload image if provided
    if image_buffer:
        try:
            image_buffer.seek(0)
            img_data = image_buffer.read()
            img_headers = headers.copy()
            img_headers.update({
                "Content-Disposition": f"attachment; filename={title.replace(' ', '_')}.jpg",
                "Content-Type": "image/jpeg"
            })
            media_url = f"{wp_config['base_url']}/wp-json/wp/v2/media"
            img_resp = requests.post(media_url, headers=img_headers, data=img_data)
            
            if img_resp.status_code == 201:
                img_id = img_resp.json()["id"]
        except Exception as e:
            st.error(f"Image upload error: {str(e)}")
    
    # Create/get tags
    tag_ids = []
    if tags:
        for tag in [t.strip() for t in tags if t.strip()]:
            try:
                tag_check = requests.get(f"{wp_config['base_url']}/wp-json/wp/v2/tags?search={tag}", headers=headers)
                if tag_check.status_code == 200 and tag_check.json():
                    tag_ids.append(tag_check.json()[0]["id"])
                else:
                    tag_create = requests.post(f"{wp_config['base_url']}/wp-json/wp/v2/tags", headers=headers, json={"name": tag})
                    if tag_create.status_code == 201:
                        tag_ids.append(tag_create.json()["id"])
            except Exception as e:
                st.warning(f"Tag creation failed for '{tag}': {str(e)}")
    
    # Publish article
    post_data = {
        "title": title,
        "content": content,
        "status": "publish" if publish_now else "draft",
        "tags": tag_ids
    }
    
    if img_id:
        post_data["featured_media"] = img_id
    
    try:
        post_resp = requests.post(f"{wp_config['base_url']}/wp-json/wp/v2/posts", headers=headers, json=post_data)
        if post_resp.status_code == 201:
            post_url = post_resp.json()["link"]
            return {"success": True, "url": post_url}
        else:
            return {"success": False, "error": post_resp.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")

# AI Provider Selection
ai_provider = st.sidebar.selectbox(
    "AI Provider",
    ["Grok (X.AI)", "OpenAI"],
    help="Select AI provider for content analysis and image prompt generation"
)

current_api_key = get_api_key(ai_provider)

if current_api_key:
    st.sidebar.success(f"✅ {ai_provider} API key found")
else:
    st.sidebar.error(f"❌ {ai_provider} API key not found")

# WordPress Configuration
st.sidebar.header("🌐 WordPress Configuration")

# Use custom config or secrets
use_custom_wp = st.sidebar.checkbox("Use Custom WordPress Settings", value=bool(st.session_state["custom_wp_config"]))

if use_custom_wp:
    with st.sidebar.form("wp_config_form"):
        st.write("**Custom WordPress Settings:**")
        wp_base_url = st.text_input("Website URL", value=st.session_state["custom_wp_config"].get("base_url", ""), placeholder="https://yoursite.com")
        wp_username = st.text_input("Username", value=st.session_state["custom_wp_config"].get("username", ""), placeholder="admin")
        wp_password = st.text_input("Application Password", value=st.session_state["custom_wp_config"].get("password", ""), placeholder="xxxx xxxx xxxx xxxx", type="password")
        
        if st.form_submit_button("💾 Save WordPress Config"):
            if wp_base_url and wp_username and wp_password:
                # Clean URL
                parsed_url = urlparse(wp_base_url)
                clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                
                st.session_state["custom_wp_config"] = {
                    "base_url": clean_url,
                    "username": wp_username,
                    "password": wp_password
                }
                st.sidebar.success("✅ WordPress config saved!")
                st.rerun()
            else:
                st.sidebar.error("❌ Please fill all fields")
    
    wp_config = st.session_state["custom_wp_config"]
else:
    # Use secrets
    wp_config = {
        "base_url": st.secrets.get("WP_BASE_URL", ""),
        "username": st.secrets.get("WP_USERNAME", ""),
        "password": st.secrets.get("WP_PASSWORD", "")
    }

# WordPress status
if wp_config.get("base_url") and wp_config.get("username") and wp_config.get("password"):
    st.sidebar.success("✅ WordPress configured")
    
    # Test connection
    if st.sidebar.button("🔍 Test WordPress Connection"):
        try:
            auth_str = f"{wp_config['username']}:{wp_config['password']}"
            auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
            headers = {"Authorization": f"Basic {auth_token}"}
            
            response = requests.get(f"{wp_config['base_url']}/wp-json/wp/v2/posts?per_page=1", headers=headers)
            if response.status_code == 200:
                st.sidebar.success("✅ WordPress connection successful!")
            else:
                st.sidebar.error(f"❌ Connection failed: {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"❌ Connection error: {str(e)}")
else:
    st.sidebar.warning("⚠️ WordPress not configured")

# Initialize HF client
hf_client = init_hf_client()
if hf_client:
    st.sidebar.success("✅ Hugging Face configured")
else:
    st.sidebar.warning("⚠️ Hugging Face not configured")

# Main App
st.title("📚 Enhanced SEO Content Automation")
st.markdown("Upload articles → Generate metadata → Create optimized images → Bulk publish to WordPress")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📤 Bulk Upload", 
    "📝 Metadata Generation", 
    "🖼️ Image Generation",
    "🔗 Internal Linking", 
    "🚀 WordPress Publish", 
    "📊 Export & Analytics",
    "⚙️ Advanced Tools"
])

with tab1:
    st.header("📤 Step 1: Bulk Article Upload")
    st.markdown("Upload up to 10 articles at once in HTML, TXT, or DOCX format")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Articles (Max 10)",
        type=['html', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Supported formats: HTML, TXT, DOCX"
    )
    
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("❌ Maximum 10 files allowed. Please select fewer files.")
        else:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
            
            # Process uploaded files
            if st.button("📋 Process Uploaded Articles"):
                processed_articles = {}
                
                for uploaded_file in uploaded_files:
                    try:
                        file_name = uploaded_file.name
                        
                        # Read file content based on type
                        if file_name.endswith('.html'):
                            content = uploaded_file.read().decode('utf-8')
                        elif file_name.endswith('.txt'):
                            content = uploaded_file.read().decode('utf-8')
                        elif file_name.endswith('.docx'):
                            doc = Document(uploaded_file)
                            content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                        
                        # Extract title
                        title = extract_title_from_content(content)
                        if not title:
                            title = file_name.replace('.html', '').replace('.txt', '').replace('.docx', '')
                        
                        processed_articles[file_name] = {
                            'title': title,
                            'content': content,
                            'file_type': file_name.split('.')[-1]
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {file_name}: {str(e)}")
                
                if processed_articles:
                    st.session_state["uploaded_articles"] = processed_articles
                    st.success(f"✅ Processed {len(processed_articles)} articles successfully!")
                    
                    # Display preview
                    for file_name, article_data in processed_articles.items():
                        with st.expander(f"📄 {file_name} - {article_data['title'][:50]}..."):
                            st.write(f"**Title:** {article_data['title']}")
                            st.write(f"**Content Preview:** {article_data['content'][:300]}...")
                            st.write(f"**File Type:** {article_data['file_type'].upper()}")
    
    # Show current uploaded articles
    if st.session_state["uploaded_articles"]:
        st.subheader(f"📋 Current Articles ({len(st.session_state['uploaded_articles'])})")
        
        for file_name, article_data in st.session_state["uploaded_articles"].items():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{article_data['title'][:60]}...** ({file_name})")
            with col2:
                if st.button("🗑️", key=f"remove_{file_name}", help="Remove article"):
                    del st.session_state["uploaded_articles"][file_name]
                    st.rerun()

with tab2:
    st.header("📝 Step 2: Metadata Generation")
    
    if st.session_state["uploaded_articles"] and current_api_key:
        st.subheader("Generate SEO Metadata for Articles")
        
        # Single article metadata
        article_files = list(st.session_state["uploaded_articles"].keys())
        selected_file = st.selectbox("Select article for metadata generation", article_files)
        
        if selected_file:
            article_data = st.session_state["uploaded_articles"][selected_file]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Title:** {article_data['title']}")
                st.write(f"**Content Preview:** {article_data['content'][:200]}...")
            
            with col2:
                if st.button("🔍 Generate Metadata"):
                    with st.spinner("Analyzing article and generating metadata..."):
                        metadata = generate_metadata_for_article(
                            article_data['content'],
                            article_data['title'],
                            current_api_key,
                            ai_provider
                        )
                        
                        if metadata:
                            st.session_state["article_metadata"][selected_file] = metadata
                            st.success("✅ Metadata generated!")
                            
                            # Display metadata
                            st.json(metadata)
        
        # Bulk metadata generation
        st.subheader("🚀 Bulk Metadata Generation")
        if st.button("📊 Generate Metadata for All Articles"):
            progress_bar = st.progress(0)
            
            for i, (file_name, article_data) in enumerate(st.session_state["uploaded_articles"].items()):
                st.info(f"Processing: {article_data['title'][:50]}...")
                
                metadata = generate_metadata_for_article(
                    article_data['content'],
                    article_data['title'],
                    current_api_key,
                    ai_provider
                )
                
                if metadata:
                    st.session_state["article_metadata"][file_name] = metadata
                
                progress_bar.progress((i + 1) / len(st.session_state["uploaded_articles"]))
                time.sleep(1.5)  # Rate limiting
            
            st.success(f"✅ Generated metadata for {len(st.session_state['article_metadata'])} articles!")
        
        # Display current metadata
        if st.session_state["article_metadata"]:
            st.subheader("📋 Generated Metadata")
            
            metadata_list = []
            for file_name, metadata in st.session_state["article_metadata"].items():
                article_title = st.session_state["uploaded_articles"][file_name]['title']
                metadata_list.append({
                    "File": file_name,
                    "Article Title": article_title[:50] + "...",
                    "SEO Title": metadata.get("seo_title", ""),
                    "Primary Keyword": metadata.get("primary_keyword", ""),
                    "Category": metadata.get("category", ""),
                    "Search Volume": metadata.get("search_volume", ""),
                    "Tags": ", ".join(metadata.get("tags", []))
                })
            
            metadata_df = pd.DataFrame(metadata_list)
            st.dataframe(metadata_df, use_container_width=True)
            
            # Export metadata
            if st.button("⬇️ Export Metadata CSV"):
                csv = metadata_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Metadata",
                    csv,
                    file_name="article_metadata.csv",
                    mime="text/csv"
                )
    
    elif not current_api_key:
        st.error(f"❌ {ai_provider} API key not found")
    else:
        st.info("⚠️ Please upload articles first in Step 1")

with tab3:
    st.header("🖼️ Step 3: AI-Optimized Image Generation")
    
    if st.session_state["uploaded_articles"] and st.session_state["article_metadata"] and hf_client and current_api_key:
        st.subheader("Generate Optimized Images")
        
        # Single image generation
        article_files = [f for f in st.session_state["uploaded_articles"].keys() if f in st.session_state["article_metadata"]]
        selected_file = st.selectbox("Select article for image generation", article_files, key="img_select")
        
        if selected_file:
            article_data = st.session_state["uploaded_articles"][selected_file]
            metadata = st.session_state["article_metadata"][selected_file]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Article:** {article_data['title']}")
                st.write(f"**Primary Keyword:** {metadata.get('primary_keyword', 'N/A')}")
                st.write(f"**Category:** {metadata.get('category', 'N/A')}")
            
            with col2:
                if st.button("🎨 Generate Optimized Image"):
                    with st.spinner("Creating optimized image prompt..."):
                        # Generate optimized prompt using AI
                        optimized_prompt = generate_optimized_image_prompt(
                            article_data['title'],
                            article_data['content'],
                            metadata.get('primary_keyword', ''),
                            current_api_key,
                            ai_provider
                        )
                        
                        st.info(f"**Generated Prompt:** {optimized_prompt}")
                        
                        # Generate image
                        with st.spinner("Generating image..."):
                            image_buffer = generate_ai_image(optimized_prompt, hf_client)
                            
                            if image_buffer:
                                st.session_state["images"][selected_file] = image_buffer
                                st.success("✅ Image generated!")
                                st.image(image_buffer, caption=f"Generated for: {article_data['title']}")
        
        # Bulk image generation
        st.subheader("🚀 Bulk Image Generation")
        if st.button("🎨 Generate Images for All Articles"):
            progress_bar = st.progress(0)
            
            for i, file_name in enumerate(article_files):
                article_data = st.session_state["uploaded_articles"][file_name]
                metadata = st.session_state["article_metadata"][file_name]
                
                st.info(f"Generating image for: {article_data['title'][:50]}...")
                
                # Generate optimized prompt
                optimized_prompt = generate_optimized_image_prompt(
                    article_data['title'],
                    article_data['content'],
                    metadata.get('primary_keyword', ''),
                    current_api_key,
                    ai_provider
                )
                
                # Generate image
                image_buffer = generate_ai_image(optimized_prompt, hf_client)
                
                if image_buffer:
                    st.session_state["images"][file_name] = image_buffer
                
                progress_bar.progress((i + 1) / len(article_files))
                time.sleep(3)  # Rate limiting for image generation
            
            st.success(f"✅ Generated {len(st.session_state['images'])} images!")
        
        # Display generated images
        if st.session_state["images"]:
            st.subheader("🖼️ Generated Images")
            
            cols = st.columns(3)
            for i, (file_name, image_buffer) in enumerate(st.session_state["images"].items()):
                with cols[i % 3]:
                    article_title = st.session_state["uploaded_articles"][file_name]['title']
                    st.image(image_buffer, caption=article_title[:30] + "...")
    
    elif not hf_client:
        st.error("❌ Hugging Face client not initialized")
    elif not current_api_key:
        st.error(f"❌ {ai_provider} API key not found")
    else:
        st.info("⚠️ Please complete Steps 1 and 2 first")

with tab4:
    st.header("🔗 Step 4: Smart Internal Linking")
    
    if wp_config.get("base_url") and wp_config.get("username") and wp_config.get("password"):
        st.subheader("Fetch Existing WordPress Posts")
        
        col1, col2 = st.columns(2)
        with col1:
            posts_per_page = st.selectbox("Posts to fetch", [50, 100, 200], index=1)
        
        with col2:
            if st.button("🔄 Fetch Posts from WordPress"):
                with st.spinner("Fetching existing posts..."):
                    existing_posts = get_wordpress_posts(wp_config, posts_per_page)
                    
                    if existing_posts:
                        st.session_state["existing_posts"] = existing_posts
                        st.success(f"✅ Fetched {len(existing_posts)} posts")
        
        # Display existing posts
        if st.session_state["existing_posts"]:
            st.subheader(f"📋 Existing Posts ({len(st.session_state['existing_posts'])})")
            
            # Create a searchable table
            posts_df = pd.DataFrame([
                {
                    "Title": post["title"],
                    "URL": post["link"],
                    "Content Preview": post["content"][:100] + "..."
                }
                for post in st.session_state["existing_posts"]
            ])
            
            st.dataframe(posts_df, use_container_width=True)
            
            # Smart internal linking
            if st.session_state["uploaded_articles"] and current_api_key:
                st.subheader("🤖 AI-Powered Internal Linking")
                
                article_files = list(st.session_state["uploaded_articles"].keys())
                selected_file = st.selectbox("Select article for internal linking", article_files, key="link_select")
                
                if selected_file:
                    article_data = st.session_state["uploaded_articles"][selected_file]
                    
                    if st.button("🔍 Find Internal Link Opportunities"):
                        with st.spinner("Analyzing content for internal linking opportunities..."):
                            links_data = find_internal_linking_opportunities(
                                article_data['content'],
                                st.session_state["existing_posts"],
                                current_api_key,
                                ai_provider
                            )
                            
                            if links_data.get("links"):
                                st.success(f"✅ Found {len(links_data['links'])} linking opportunities!")
                                
                                # Display suggested links
                                for i, link in enumerate(links_data["links"]):
                                    with st.expander(f"🔗 Link {i+1}: {link['anchor_text']}"):
                                        st.write(f"**Anchor Text:** {link['anchor_text']}")
                                        st.write(f"**Target Post:** {link['target_post']}")
                                        st.write(f"**URL:** {link['target_url']}")
                                        st.write(f"**Reason:** {link['reason']}")
                                
                                # Apply links
                                if st.button("✅ Apply Internal Links"):
                                    modified_content = apply_internal_links_to_content(
                                        article_data['content'],
                                        links_data
                                    )
                                    
                                    # Update the article content
                                    st.session_state["uploaded_articles"][selected_file]['content'] = modified_content
                                    st.success("✅ Internal links applied to article!")
                            else:
                                st.info("No relevant internal linking opportunities found.")
                
                # Bulk internal linking
                st.subheader("🚀 Bulk Internal Linking")
                if st.button("🔗 Apply Internal Links to All Articles"):
                    progress_bar = st.progress(0)
                    
                    for i, (file_name, article_data) in enumerate(st.session_state["uploaded_articles"].items()):
                        st.info(f"Processing internal links for: {article_data['title'][:50]}...")
                        
                        links_data = find_internal_linking_opportunities(
                            article_data['content'],
                            st.session_state["existing_posts"],
                            current_api_key,
                            ai_provider
                        )
                        
                        if links_data.get("links"):
                            modified_content = apply_internal_links_to_content(
                                article_data['content'],
                                links_data
                            )
                            st.session_state["uploaded_articles"][file_name]['content'] = modified_content
                        
                        progress_bar.progress((i + 1) / len(st.session_state["uploaded_articles"]))
                        time.sleep(2)  # Rate limiting
                    
                    st.success("✅ Internal linking completed for all articles!")
        else:
            st.info("Click 'Fetch Posts from WordPress' to load existing posts for internal linking.")
    
    else:
        st.error("❌ WordPress not configured. Please configure in the sidebar.")

with tab5:
    st.header("🚀 Step 5: WordPress Bulk Publishing")
    
    if (st.session_state["uploaded_articles"] and 
        st.session_state["article_metadata"] and 
        wp_config.get("base_url") and 
        wp_config.get("username") and 
        wp_config.get("password")):
        
        st.subheader("📤 Publishing Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            publish_mode = st.radio("Publishing Mode", ["Draft", "Publish Immediately"])
            publish_now = publish_mode == "Publish Immediately"
        
        with col2:
            global_tags = st.text_input("Additional Tags (comma-separated)", "education,india,guide")
        
        # Single article publishing
        st.subheader("📝 Publish Single Article")
        
        ready_articles = [f for f in st.session_state["uploaded_articles"].keys() if f in st.session_state["article_metadata"]]
        selected_file = st.selectbox("Select article to publish", ready_articles, key="publish_select")
        
        if selected_file:
            article_data = st.session_state["uploaded_articles"][selected_file]
            metadata = st.session_state["article_metadata"][selected_file]
            
            # Preview publishing details
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Title:** {metadata.get('seo_title', article_data['title'])}")
                st.write(f"**Primary Keyword:** {metadata.get('primary_keyword', '')}")
                st.write(f"**Has Image:** {'✅' if selected_file in st.session_state['images'] else '❌'}")
            
            with col2:
                all_tags = metadata.get('tags', [])
                if global_tags:
                    all_tags.extend([tag.strip() for tag in global_tags.split(',') if tag.strip()])
                st.write(f"**Tags:** {', '.join(all_tags)}")
                st.write(f"**Status:** {publish_mode}")
            
            if st.button("📤 Publish Selected Article"):
                with st.spinner("Publishing to WordPress..."):
                    result = publish_to_wordpress(
                        metadata.get('seo_title', article_data['title']),
                        article_data['content'],
                        st.session_state["images"].get(selected_file),
                        all_tags,
                        wp_config,
                        publish_now
                    )
                    
                    if result["success"]:
                        st.success(f"✅ Published successfully!")
                        st.success(f"🔗 **URL:** {result['url']}")
                        
                        # Log the publication
                        st.session_state["publish_log"].append({
                            "article": metadata.get('seo_title', article_data['title']),
                            "url": result["url"],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "Published" if publish_now else "Draft"
                        })
                    else:
                        st.error(f"❌ Publishing failed: {result['error']}")
        
        # Bulk publishing
        st.subheader("🚀 Bulk Publishing")
        
        if ready_articles:
            st.info(f"Ready to publish: {len(ready_articles)} articles")
            
            # Show what will be published
            with st.expander("📋 Preview Articles to Publish"):
                for file_name in ready_articles:
                    article_data = st.session_state["uploaded_articles"][file_name]
                    metadata = st.session_state["article_metadata"][file_name]
                    has_image = "✅" if file_name in st.session_state["images"] else "❌"
                    
                    st.write(f"**{metadata.get('seo_title', article_data['title'])}** - Image: {has_image}")
            
            if st.button("🚀 Publish All Articles"):
                progress_bar = st.progress(0)
                success_count = 0
                
                for i, file_name in enumerate(ready_articles):
                    article_data = st.session_state["uploaded_articles"][file_name]
                    metadata = st.session_state["article_metadata"][file_name]
                    
                    st.info(f"Publishing: {metadata.get('seo_title', article_data['title'])[:50]}...")
                    
                    # Prepare tags
                    all_tags = metadata.get('tags', [])
                    if global_tags:
                        all_tags.extend([tag.strip() for tag in global_tags.split(',') if tag.strip()])
                    
                    # Publish
                    result = publish_to_wordpress(
                        metadata.get('seo_title', article_data['title']),
                        article_data['content'],
                        st.session_state["images"].get(file_name),
                        all_tags,
                        wp_config,
                        publish_now
                    )
                    
                    if result["success"]:
                        success_count += 1
                        st.success(f"✅ Published: {metadata.get('seo_title', article_data['title'])[:30]}...")
                        
                        # Log publication
                        st.session_state["publish_log"].append({
                            "article": metadata.get('seo_title', article_data['title']),
                            "url": result["url"],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "Published" if publish_now else "Draft"
                        })
                    else:
                        st.error(f"❌ Failed: {metadata.get('seo_title', article_data['title'])[:30]}...")
                    
                    progress_bar.progress((i + 1) / len(ready_articles))
                    time.sleep(2)  # Rate limiting
                
                st.success(f"🎉 Bulk publishing completed! {success_count}/{len(ready_articles)} articles published successfully.")
        else:
            st.warning("⚠️ No articles ready for publishing. Complete Steps 1-2 first.")
    
    else:
        missing = []
        if not st.session_state["uploaded_articles"]:
            missing.append("uploaded articles")
        if not st.session_state["article_metadata"]:
            missing.append("article metadata")
        if not wp_config.get("base_url"):
            missing.append("WordPress configuration")
        
        st.error(f"❌ Missing: {', '.join(missing)}")

with tab6:
    st.header("📊 Step 6: Export & Analytics")
    
    # Publication Log
    if st.session_state["publish_log"]:
        st.subheader("📈 Publication Log")
        
        log_df = pd.DataFrame(st.session_state["publish_log"])
        st.dataframe(log_df, use_container_width=True)
        
        # Analytics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Published", len(log_df))
        with col2:
            published_count = len(log_df[log_df["status"] == "Published"])
            st.metric("Live Articles", published_count)
        with col3:
            draft_count = len(log_df[log_df["status"] == "Draft"])
            st.metric("Drafts", draft_count)
        
        # Export log
        log_csv = log_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Publication Log",
            log_csv,
            file_name=f"publication_log_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Export Articles
    if st.session_state["uploaded_articles"]:
        st.subheader("📦 Export Articles")
        
        # Individual exports
        article_files = list(st.session_state["uploaded_articles"].keys())
        selected_file = st.selectbox("Select article to export", article_files, key="export_select")
        
        if selected_file:
            article_data = st.session_state["uploaded_articles"][selected_file]
            
            col1, col2 = st.columns(2)
            with col1:
                # HTML export
                st.download_button(
                    "⬇️ Download HTML",
                    article_data['content'],
                    file_name=f"{article_data['title'][:30].replace(' ', '_')}.html",
                    mime="text/html"
                )
            
            with col2:
                # Text export
                clean_content = re.sub(r'<[^>]+>', '', article_data['content'])
                st.download_button(
                    "⬇️ Download Text",
                    clean_content,
                    file_name=f"{article_data['title'][:30].replace(' ', '_')}.txt",
                    mime="text/plain"
                )
        
        # Bulk export
        if st.button("📦 Create Complete Export Package"):
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add articles
                for file_name, article_data in st.session_state["uploaded_articles"].items():
                    # HTML version
                    zip_file.writestr(f"articles/{article_data['title'][:30].replace(' ', '_')}.html", article_data['content'])
                    
                    # Text version
                    clean_content = re.sub(r'<[^>]+>', '', article_data['content'])
                    zip_file.writestr(f"articles_text/{article_data['title'][:30].replace(' ', '_')}.txt", clean_content)
                
                # Add metadata
                if st.session_state["article_metadata"]:
                    metadata_list = []
                    for file_name, metadata in st.session_state["article_metadata"].items():
                        article_title = st.session_state["uploaded_articles"][file_name]['title']
                        metadata_list.append({
                            "File": file_name,
                            "Article Title": article_title,
                            "SEO Title": metadata.get("seo_title", ""),
                            "Meta Description": metadata.get("meta_description", ""),
                            "Primary Keyword": metadata.get("primary_keyword", ""),
                            "Secondary Keywords": ", ".join(metadata.get("secondary_keywords", [])),
                            "Category": metadata.get("category", ""),
                            "Search Volume": metadata.get("search_volume", ""),
                            "Tags": ", ".join(metadata.get("tags", []))
                        })
                    
                    metadata_df = pd.DataFrame(metadata_list)
                    metadata_csv = metadata_df.to_csv(index=False)
                    zip_file.writestr("metadata/article_metadata.csv", metadata_csv)
                
                # Add publication log
                if st.session_state["publish_log"]:
                    log_df = pd.DataFrame(st.session_state["publish_log"])
                    log_csv = log_df.to_csv(index=False)
                    zip_file.writestr("logs/publication_log.csv", log_csv)
                
                # Add images
                for file_name, image_buffer in st.session_state["images"].items():
                    article_title = st.session_state["uploaded_articles"][file_name]['title']
                    image_buffer.seek(0)
                    zip_file.writestr(f"images/{article_title[:30].replace(' ', '_')}.png", image_buffer.read())
            
            zip_buffer.seek(0)
            
            st.download_button(
                "⬇️ Download Complete Package (ZIP)",
                zip_buffer,
                file_name=f"seo_content_package_{time.strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip"
            )

with tab7:
    st.header("⚙️ Advanced Tools & Settings")
    
    # Data Management
    st.subheader("🗃️ Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Clear All Articles"):
            st.session_state["uploaded_articles"] = {}
            st.session_state["article_metadata"] = {}
            st.session_state["images"] = {}
            st.success("✅ All articles cleared!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Publication Log"):
            st.session_state["publish_log"] = []
            st.success("✅ Publication log cleared!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Everything"):
            for key in list(st.session_state.keys()):
                if key != "custom_wp_config":  # Keep WordPress config
                    del st.session_state[key]
            st.success("✅ Everything reset!")
            st.rerun()
    
    # System Status
    st.subheader("📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Uploaded Articles", len(st.session_state.get("uploaded_articles", {})))
    
    with col2:
        st.metric("Generated Metadata", len(st.session_state.get("article_metadata", {})))
    
    with col3:
        st.metric("Generated Images", len(st.session_state.get("images", {})))
    
    with col4:
        st.metric("Published Articles", len(st.session_state.get("publish_log", [])))
    
    # API Configuration Help
    st.subheader("🔑 API Configuration Help")
    
    with st.expander("📋 Required API Keys & Configuration"):
        st.markdown("""
        **Required Secrets in Streamlit Cloud:**
        
        ```toml
        # AI Provider (choose one)
        GROK_API_KEY = "xai-..."
        OPENAI_API_KEY = "sk-..."
        
        # Image Generation
        HF_TOKEN = "hf_..."
        
        # WordPress (optional, can use custom config)
        WP_BASE_URL = "https://yoursite.com"
        WP_USERNAME = "admin"
        WP_PASSWORD = "xxxx xxxx xxxx xxxx"
        ```
        
        **How to get API keys:**
        - **Grok API**: Visit https://x.ai/api
        - **OpenAI API**: Visit https://platform.openai.com/api-keys
        - **Hugging Face**: Visit https://huggingface.co/settings/tokens
        - **WordPress**: Create Application Password in WordPress admin
        """)
    
    # Workflow Tips
    st.subheader("💡 Workflow Tips")
    
    with st.expander("🚀 Optimization Tips"):
        st.markdown("""
        **For Best Results:**
        
        1. **Article Upload**: Use well-formatted HTML files for best metadata extraction
        2. **File Naming**: Use descriptive filenames that reflect the article topic
        3. **Content Quality**: Ensure articles are complete with proper headings and structure
        4. **Rate Limiting**: The system includes automatic delays to respect API limits
        5. **Image Generation**: AI-optimized prompts work better than generic descriptions
        6. **Internal Linking**: Fetch existing posts before running bulk operations
        7. **WordPress**: Test connection before bulk publishing
        8. **Backup**: Always export your work before major operations
        
        **Troubleshooting:**
        - If API calls fail, check your keys and try a different provider
        - For WordPress issues, verify your application password is correct
        - Large images may take longer to upload to WordPress
        - Some WordPress themes may require specific image formats
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📚 Enhanced SEO Content Automation Pipeline | Built with Streamlit</p>
    <p>Upload → Analyze → Optimize → Link → Publish | Streamlined content workflow for any WordPress site</p>
</div>
""", unsafe_allow_html=True)
