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
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
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
if "existing_tags" not in st.session_state:
    st.session_state["existing_tags"] = []
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

def get_wordpress_tags(wp_config):
    """Fetch existing WordPress tags"""
    try:
        auth_str = f"{wp_config['username']}:{wp_config['password']}"
        auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
        headers = {
            "Authorization": f"Basic {auth_token}",
            "Content-Type": "application/json"
        }
        
        wp_base = wp_config['base_url'].rstrip('/')
        url = f"{wp_base}/wp-json/wp/v2/tags?per_page=100"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            tags_data = response.json()
            return [{"id": tag["id"], "name": tag["name"], "slug": tag["slug"]} for tag in tags_data]
        else:
            st.warning(f"Failed to fetch tags: HTTP {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Error fetching tags: {str(e)}")
        return []

def create_or_get_tags(tag_names, wp_config):
    """Create new tags or get existing ones"""
    auth_str = f"{wp_config['username']}:{wp_config['password']}"
    auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
    headers = {
        "Authorization": f"Basic {auth_token}",
        "Content-Type": "application/json"
    }
    
    wp_base = wp_config['base_url'].rstrip('/')
    tag_ids = []
    
    for tag_name in tag_names:
        if not tag_name.strip():
            continue
            
        tag_name = tag_name.strip()
        
        try:
            # Check if tag already exists
            search_url = f"{wp_base}/wp-json/wp/v2/tags"
            search_params = {"search": tag_name}
            search_response = requests.get(search_url, headers=headers, params=search_params, timeout=10)
            
            if search_response.status_code == 200:
                search_results = search_response.json()
                # Look for exact match
                for result in search_results:
                    if result["name"].lower() == tag_name.lower():
                        tag_ids.append(result["id"])
                        break
                else:
                    # Create new tag
                    create_url = f"{wp_base}/wp-json/wp/v2/tags"
                    tag_data = {"name": tag_name}
                    
                    create_response = requests.post(create_url, headers=headers, json=tag_data, timeout=10)
                    
                    if create_response.status_code == 201:
                        new_tag = create_response.json()
                        tag_ids.append(new_tag["id"])
        
        except Exception as e:
            continue  # Skip this tag on error
    
    return tag_ids

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
7. Tags for WordPress (5-8 relevant SEO tags)

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
                    content_response = response.json()["choices"][0]["message"]["content"]
                    try:
                        return json.loads(content_response)
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
                content_response = response.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(content_response)
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

Create a concise, effective prompt (max 50 words) that will create a high-quality, professional image suitable for a blog article header.

Return only the prompt text, no quotes or explanations.
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

def create_property_overlay(img, primary_text, secondary_text, output_size=(1200, 675)):
    """Create a property overlay with responsive design using MagicBricks style"""
    
    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to target size
    img = img.resize(output_size, Image.Resampling.LANCZOS)
    width, height = output_size
    
    # Create overlay layer
    overlay = Image.new("RGBA", output_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Color scheme - vibrant red like MagicBricks
    gradient_red = (231, 76, 60)
    gradient_dark = (192, 57, 43)
    white = (255, 255, 255)
    
    # Determine layout based on aspect ratio
    aspect_ratio = width / height
    
    if aspect_ratio > 1.5:  # Wide format (landscape)
        text_area_percentage = 0.30
        gradient_position = "bottom"
    elif aspect_ratio < 0.8:  # Tall format (portrait/mobile)
        text_area_percentage = 0.25
        gradient_position = "bottom"
    else:  # Square or near-square
        text_area_percentage = 0.25
        gradient_position = "bottom"
    
    # Calculate gradient dimensions
    if gradient_position == "bottom":
        gradient_height = int(height * text_area_percentage)
        gradient_start_y = height - gradient_height
        gradient_width = width
        gradient_start_x = 0
    
    # Create completely dissolved/seamless gradient
    for y in range(gradient_height):
        # Calculate gradient progression (0 to 1)
        progress = y / gradient_height
        
        # Ultra-smooth easing for completely dissolved effect
        eased_progress = progress * progress * progress * (progress * (progress * 6 - 15) + 10)
        
        # Very gradual alpha progression for dissolved effect
        alpha = int(15 + (200 * eased_progress))
        
        # Smooth color interpolation
        r = int(gradient_dark[0] + (gradient_red[0] - gradient_dark[0]) * eased_progress)
        g = int(gradient_dark[1] + (gradient_red[1] - gradient_dark[1]) * eased_progress)
        b = int(gradient_dark[2] + (gradient_red[2] - gradient_dark[2]) * eased_progress)
        
        # Draw ultra-smooth gradient line
        draw.rectangle([gradient_start_x, gradient_start_y + y, 
                       gradient_start_x + gradient_width, gradient_start_y + y + 1], 
                      fill=(r, g, b, alpha))
    
    # Add additional smoothing blur effect above gradient
    blur_height = max(20, int(height * 0.02))  # Responsive blur height
    for i in range(blur_height):
        progress = 1 - (i / blur_height)
        alpha = int(5 * progress)
        draw.rectangle([gradient_start_x, gradient_start_y - blur_height + i, 
                       gradient_start_x + gradient_width, gradient_start_y - blur_height + i + 1], 
                      fill=(gradient_dark[0], gradient_dark[1], gradient_dark[2], alpha))
    
    # Responsive font sizing based on image dimensions
    def get_responsive_font_size(base_size, width, height):
        scale_factor = min(width / 1200, height / 675)  # Base reference size
        return max(12, int(base_size * scale_factor))
    
    # Load fonts with marketing-style emphasis
    def get_marketing_font(size, bold=False):
        try:
            return ImageFont.load_default()
        except:
            return ImageFont.load_default()
    
    # Responsive font sizes
    primary_font_size = get_responsive_font_size(36, width, height)
    secondary_font_size = get_responsive_font_size(24, width, height)
    
    primary_font = get_marketing_font(primary_font_size, bold=True)
    secondary_font = get_marketing_font(secondary_font_size, bold=False)
    
    # Responsive text positioning
    text_padding = max(20, int(width * 0.025))
    text_start_y = gradient_start_y + (gradient_height * 0.6)
    
    # Helper function for marketing-style text with strong shadows
    def draw_marketing_text(text, position, font, text_color=white):
        x, y = position
        
        # Responsive shadow offsets
        shadow_scale = min(width / 1200, height / 675)
        shadow_offsets = [
            (int(2 * shadow_scale), int(2 * shadow_scale)),
            (int(1 * shadow_scale), int(1 * shadow_scale)),
            (int(3 * shadow_scale), int(3 * shadow_scale)),
            (int(4 * shadow_scale), int(4 * shadow_scale))
        ]
        
        for offset_x, offset_y in shadow_offsets:
            shadow_alpha = max(50, 150 - (offset_x * 30))
            draw.text((x + offset_x, y + offset_y), text, font=font, fill=(0, 0, 0, shadow_alpha))
        
        # Main text
        draw.text((x, y), text, font=font, fill=text_color)
    
    # Position text with word wrapping for smaller sizes
    def wrap_text(text, font, max_width):
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)  # Single word too long
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    # Text positioning with responsive wrapping
    available_text_width = width - (text_padding * 2)
    
    if primary_text:
        # Wrap primary text if needed
        primary_lines = wrap_text(primary_text, primary_font, available_text_width)
        
        # Position primary text
        current_y = text_start_y
        for line in primary_lines:
            bbox = draw.textbbox((0, 0), line, font=primary_font)
            line_width = bbox[2] - bbox[0]
            line_x = (width - line_width) // 2
            
            draw_marketing_text(line, (line_x, current_y), primary_font)
            current_y += (bbox[3] - bbox[1]) + 5
        
        if secondary_text:
            # Wrap secondary text if needed
            secondary_lines = wrap_text(secondary_text, secondary_font, available_text_width)
            
            # Position secondary text below primary
            current_y += 10  # Gap between primary and secondary
            
            for line in secondary_lines:
                bbox = draw.textbbox((0, 0), line, font=secondary_font)
                line_width = bbox[2] - bbox[0]
                line_x = (width - line_width) // 2
                
                draw_marketing_text(line, (line_x, current_y), secondary_font)
                current_y += (bbox[3] - bbox[1]) + 5
    
    # Composite the overlay onto the image
    final_img = Image.alpha_composite(img.convert("RGBA"), overlay)
    
    return final_img.convert("RGB")

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
        headers = {
            "Authorization": f"Basic {auth_token}",
            "Content-Type": "application/json"
        }
        
        wp_base = wp_config['base_url'].rstrip('/')
        url = f"{wp_base}/wp-json/wp/v2/posts?per_page={per_page}&status=publish"
        response = requests.get(url, headers=headers, timeout=15)
        
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

def prepare_article_html(content, metadata):
    """Convert article content to proper HTML with metadata"""
    
    # Clean and structure the content
    html_content = content
    
    # Ensure basic HTML structure
    if not html_content.strip().startswith('<'):
        # Convert plain text to HTML
        html_content = f"<div>{html_content}</div>"
    
    # Add meta description as hidden comment for SEO
    if metadata and metadata.get('meta_description'):
        meta_comment = f"<!-- Meta Description: {metadata['meta_description']} -->\n"
        html_content = meta_comment + html_content
    
    # Add primary keyword as hidden comment
    if metadata and metadata.get('primary_keyword'):
        keyword_comment = f"<!-- Primary Keyword: {metadata['primary_keyword']} -->\n"
        html_content = keyword_comment + html_content
    
    # Ensure proper paragraph structure
    if '<p>' not in html_content and '<div>' not in html_content:
        # Wrap content in paragraphs
        paragraphs = html_content.split('\n\n')
        html_content = ''.join([f"<p>{p.strip()}</p>\n" for p in paragraphs if p.strip()])
    
    return html_content

def publish_to_wordpress(title, content, metadata, image_buffer, wp_config, publish_now=True):
    """Publish article to WordPress - completely fixed version with better error handling"""
    try:
        wp_base = wp_config["base_url"].rstrip('/')
        auth_str = f"{wp_config['username']}:{wp_config['password']}"
        auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
        
        # Standard headers for all requests
        headers = {
            "Authorization": f"Basic {auth_token}",
            "Content-Type": "application/json"
        }
        
        # Use SEO title if available
        final_title = metadata.get('seo_title', title) if metadata else title
        
        # Prepare content
        html_content = prepare_article_html(content, metadata) if metadata else content
        
        img_id = None
        
        # Upload image if provided
        if image_buffer:
            try:
                image_buffer.seek(0)
                img_data = image_buffer.read()
                
                # Specific headers for image upload
                img_headers = {
                    "Authorization": f"Basic {auth_token}",
                    "Content-Disposition": f"attachment; filename={final_title.replace(' ', '_')}.jpg",
                    "Content-Type": "image/jpeg"
                }
                
                media_url = f"{wp_base}/wp-json/wp/v2/media"
                img_resp = requests.post(media_url, headers=img_headers, data=img_data, timeout=30)
                
                if img_resp.status_code == 201:
                    img_id = img_resp.json()["id"]
                else:
                    st.warning(f"Image upload failed: HTTP {img_resp.status_code}")
                    
            except Exception as e:
                st.warning(f"Image upload error: {str(e)}")
                # Continue without image
        
        # Create/get tags
        tag_ids = []
        if metadata and metadata.get('tags'):
            try:
                tag_ids = create_or_get_tags(metadata['tags'], wp_config)
            except Exception as e:
                st.warning(f"Tag creation failed: {str(e)}")
                # Continue without tags
        
        # Prepare post data
        post_data = {
            "title": final_title,
            "content": html_content,
            "status": "publish" if publish_now else "draft"
        }
        
        if tag_ids:
            post_data["tags"] = tag_ids
        
        if img_id:
            post_data["featured_media"] = img_id
        
        if metadata and metadata.get('meta_description'):
            post_data["excerpt"] = metadata['meta_description']
        
        # Test REST API endpoint first
        test_url = f"{wp_base}/wp-json/wp/v2"
        test_resp = requests.get(test_url, headers=headers, timeout=20)
        
        if test_resp.status_code != 200:
            return {"success": False, "error": f"REST API not accessible: HTTP {test_resp.status_code}"}
        
        # Check if response is JSON (not HTML)
        try:
            test_resp.json()
        except Exception as e:
            return {"success": False, "error": f"REST API returned invalid JSON: {str(e)}"}
        
        # Now try to publish
        post_url = f"{wp_base}/wp-json/wp/v2/posts"
        post_resp = requests.post(post_url, headers=headers, json=post_data, timeout=30)
        
        # Enhanced error handling for different response types
        content_type = post_resp.headers.get('content-type', '').lower()
        
        if 'text/html' in content_type:
            return {"success": False, "error": "WordPress returned HTML page instead of JSON. Check authentication and REST API access."}
        
        if post_resp.status_code == 201:
            try:
                response_data = post_resp.json()
                article_url = response_data.get("link", "")
                return {"success": True, "url": article_url, "title": final_title}
            except Exception as e:
                return {"success": False, "error": f"Published but couldn't parse response: {str(e)}"}
                
        elif post_resp.status_code == 500:
            return {"success": False, "error": "WordPress Internal Server Error (500). Check: 1) Application password format, 2) Security plugins blocking REST API, 3) Plugin conflicts"}
            
        elif post_resp.status_code == 401:
            return {"success": False, "error": "Authentication failed (401). Check username and application password"}
            
        elif post_resp.status_code == 403:
            return {"success": False, "error": "Access forbidden (403). Check user permissions for publishing posts"}
            
        else:
            try:
                error_data = post_resp.json()
                error_message = error_data.get('message', 'Unknown error')
                return {"success": False, "error": f"HTTP {post_resp.status_code}: {error_message}"}
            except:
                return {"success": False, "error": f"HTTP {post_resp.status_code}: {post_resp.text[:200]}"}
    
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout - WordPress server is slow or unresponsive"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection error - check WordPress URL and network connectivity"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

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
        
        wp_base_url = st.text_input(
            "Website URL", 
            value=st.session_state["custom_wp_config"].get("base_url", ""), 
            placeholder="https://yoursite.com",
            help="Your WordPress website URL (without /wp-admin)"
        )
        
        wp_username = st.text_input(
            "Username", 
            value=st.session_state["custom_wp_config"].get("username", ""), 
            placeholder="admin",
            help="Your WordPress admin username"
        )
        
        wp_password = st.text_input(
            "Application Password", 
            value=st.session_state["custom_wp_config"].get("password", ""), 
            placeholder="xxxx xxxx xxxx xxxx", 
            type="password",
            help="WordPress Application Password (not your regular password)"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("💾 Save Config"):
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
        
        with col2:
            if st.form_submit_button("🧹 Clear Config"):
                st.session_state["custom_wp_config"] = {}
                st.sidebar.success("✅ Config cleared!")
                st.rerun()
    
    wp_config = st.session_state["custom_wp_config"]
else:
    # Use secrets
    wp_config = {
        "base_url": st.secrets.get("WP_BASE_URL", ""),
        "username": st.secrets.get("WP_USERNAME", ""),
        "password": st.secrets.get("WP_PASSWORD", "")
    }

# WordPress status and testing
if wp_config.get("base_url") and wp_config.get("username") and wp_config.get("password"):
    st.sidebar.success("✅ WordPress configured")
    
    # Show basic info
    with st.sidebar.expander("📋 WordPress Details"):
        st.write(f"**URL:** {wp_config['base_url']}")
        st.write(f"**User:** {wp_config['username']}")
        st.write(f"**Password:** {'*' * len(wp_config['password'])}")
    
    # Enhanced testing buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔍 Quick Test"):
            try:
                wp_base = wp_config["base_url"].rstrip('/')
                auth_str = f"{wp_config['username']}:{wp_config['password']}"
                auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
                headers = {
                    "Authorization": f"Basic {auth_token}",
                    "Content-Type": "application/json"
                }
                
                # Test REST API root
                test_url = f"{wp_base}/wp-json/wp/v2"
                response = requests.get(test_url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    # Check if response is JSON
                    try:
                        response.json()
                        st.sidebar.success("✅ REST API OK!")
                    except:
                        st.sidebar.error("❌ REST API returns HTML")
                elif response.status_code == 401:
                    st.sidebar.error("❌ Auth failed")
                elif response.status_code == 403:
                    st.sidebar.error("❌ Access denied")
                elif response.status_code == 404:
                    st.sidebar.error("❌ REST API not found")
                else:
                    st.sidebar.error(f"❌ Error: {response.status_code}")
            
            except requests.exceptions.Timeout:
                st.sidebar.error("❌ Timeout")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)[:20]}...")
    
    with col2:
        if st.button("📊 Full Test"):
            # This will be handled in the main app area for detailed output
            st.session_state["run_detailed_test"] = True
            st.rerun()
    
    # Add 500 Error Helper Button
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🛠️ Fix 500"):
            st.session_state["show_500_fix"] = True
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Auth"):
            st.session_state["show_auth_reset"] = True
            st.rerun()

else:
    st.sidebar.warning("⚠️ WordPress not configured")
    
    if not wp_config.get("base_url"):
        st.sidebar.error("Missing: Website URL")
    if not wp_config.get("username"):
        st.sidebar.error("Missing: Username")
    if not wp_config.get("password"):
        st.sidebar.error("Missing: Password")

# Initialize HF client
hf_client = init_hf_client()
if hf_client:
    st.sidebar.success("✅ Hugging Face configured")
else:
    st.sidebar.warning("⚠️ Hugging Face not configured")

def test_wordpress_connection_detailed(wp_config):
    """Enhanced WordPress connection testing with specific 500 error diagnosis"""
    wp_base = wp_config["base_url"].rstrip('/')
    auth_str = f"{wp_config['username']}:{wp_config['password']}"
    auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
    
    st.write("**Testing WordPress Connection...**")
    
    # Test 1: Basic connectivity
    st.write("🔍 **Test 1: Basic Website Access**")
    try:
        response = requests.get(wp_base, timeout=15)
        if response.status_code == 200:
            st.success(f"✅ Website accessible (HTTP {response.status_code})")
        else:
            st.error(f"❌ Website returned HTTP {response.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ Cannot reach website: {str(e)}")
        return False
    
    # Test 2: REST API discovery without auth
    st.write("🔍 **Test 2: REST API Discovery (No Auth)**")
    try:
        response = requests.get(f"{wp_base}/wp-json/", timeout=15)
        if response.status_code == 200:
            try:
                api_data = response.json()
                st.success("✅ REST API endpoint found")
                st.write(f"WordPress version: {api_data.get('generator', 'Unknown')}")
            except:
                st.error("❌ REST API returns invalid JSON")
                return False
        else:
            st.error(f"❌ REST API not accessible (HTTP {response.status_code})")
            return False
    except Exception as e:
        st.error(f"❌ REST API error: {str(e)}")
        return False
    
    # Test 3: Permalink structure check
    st.write("🔍 **Test 3: Permalink Structure Check**")
    try:
        # Test both permalink formats
        rest_formats = [
            f"{wp_base}/wp-json/wp/v2/",
            f"{wp_base}/?rest_route=/wp/v2/"
        ]
        
        permalink_working = False
        for rest_url in rest_formats:
            try:
                response = requests.get(rest_url, timeout=10)
                if response.status_code == 200:
                    st.success(f"✅ Permalink working: {rest_url}")
                    permalink_working = True
                    break
            except:
                continue
        
        if not permalink_working:
            st.error("❌ Neither permalink format working - check WordPress permalink settings")
            st.markdown("**Fix:** Go to WordPress Admin → Settings → Permalinks → Save Changes")
            return False
            
    except Exception as e:
        st.warning(f"⚠️ Permalink test inconclusive: {str(e)}")
    
    # Test 4: Authentication with multiple header formats
    st.write("🔍 **Test 4: Authentication Test (Multiple Formats)**")
    
    auth_tests = [
        {
            "name": "Standard Basic Auth",
            "headers": {
                "Authorization": f"Basic {auth_token}",
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Basic Auth without Content-Type",
            "headers": {
                "Authorization": f"Basic {auth_token}"
            }
        },
        {
            "name": "Username:Password format",
            "auth": (wp_config['username'], wp_config['password']),
            "headers": {"Content-Type": "application/json"}
        }
    ]
    
    auth_success = False
    for test in auth_tests:
        try:
            st.write(f"Testing: {test['name']}")
            
            if 'auth' in test:
                response = requests.get(
                    f"{wp_base}/wp-json/wp/v2/users/me", 
                    auth=test['auth'],
                    headers=test.get('headers', {}),
                    timeout=15
                )
            else:
                response = requests.get(
                    f"{wp_base}/wp-json/wp/v2/users/me", 
                    headers=test['headers'],
                    timeout=15
                )
            
            if response.status_code == 200:
                try:
                    user_data = response.json()
                    st.success(f"✅ Authentication successful with {test['name']}")
                    st.write(f"Logged in as: {user_data.get('name', 'Unknown')} (ID: {user_data.get('id', 'Unknown')})")
                    st.write(f"User roles: {', '.join(user_data.get('roles', []))}")
                    auth_success = True
                    break
                except:
                    st.error(f"❌ {test['name']}: Authentication response invalid")
            elif response.status_code == 500:
                st.error(f"❌ {test['name']}: Server error (HTTP 500)")
                # Log the actual error response for debugging
                st.code(f"Response: {response.text[:300]}")
            elif response.status_code == 401:
                st.error(f"❌ {test['name']}: Authentication failed (HTTP 401)")
            elif response.status_code == 403:
                st.error(f"❌ {test['name']}: Access forbidden (HTTP 403)")
            else:
                st.error(f"❌ {test['name']}: HTTP {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ {test['name']}: {str(e)}")
    
    if not auth_success:
        st.error("❌ All authentication methods failed")
        
        # Detailed 500 error troubleshooting
        st.subheader("🛠️ HTTP 500 Error Troubleshooting")
        st.markdown("""
        **Your site is returning HTTP 500 errors. Here's how to fix it:**
        
        **1. Application Password Format:**
        - Go to WordPress Admin → Users → Your Profile
        - Delete ALL existing application passwords
        - Create a NEW application password
        - Copy the EXACT password including spaces (e.g., `abcd efgh ijkl mnop`)
        - Do NOT remove spaces or modify the password
        
        **2. Security Plugin Check:**
        - Temporarily deactivate WordFence, Sucuri, or any security plugins
        - Check if the error persists
        - Look for "REST API" blocking in plugin settings
        
        **3. Plugin Conflicts:**
        - Deactivate ALL plugins temporarily
        - Test the REST API again
        - If it works, reactivate plugins one by one to find the culprit
        
        **4. Server Configuration:**
        - Contact your hosting provider (looks like you're using a hosting service)
        - Ask them to check for REST API blocking
        - Ensure mod_rewrite is enabled
        
        **5. WordPress Core Issue:**
        - Update WordPress to the latest version
        - Check if your theme supports REST API
        
        **6. .htaccess File:**
        - Backup your .htaccess file
        - Delete it temporarily
        - Go to WordPress Admin → Settings → Permalinks → Save
        - Test again
        """)
        return False
    
    # Test 5: Posts endpoint access
    st.write("🔍 **Test 5: Posts Endpoint Access**")
    try:
        headers = {
            "Authorization": f"Basic {auth_token}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{wp_base}/wp-json/wp/v2/posts?per_page=1", headers=headers, timeout=15)
        if response.status_code == 200:
            try:
                posts_data = response.json()
                st.success("✅ Posts endpoint accessible")
                st.write(f"Found {len(posts_data)} posts in response")
            except:
                st.error("❌ Posts endpoint returns invalid JSON")
                return False
        else:
            st.error(f"❌ Posts endpoint error (HTTP {response.status_code})")
            return False
    except Exception as e:
        st.error(f"❌ Posts endpoint test error: {str(e)}")
        return False
    
    # Test 6: Create permission test with better error handling
    st.write("🔍 **Test 6: Create Post Permission Test**")
    try:
        test_post_data = {
            "title": "REST API Test - Please Delete",
            "content": "This is a test post created by the REST API. Please delete this post.",
            "status": "draft"
        }
        
        headers = {
            "Authorization": f"Basic {auth_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(f"{wp_base}/wp-json/wp/v2/posts", 
                               headers=headers, 
                               json=test_post_data, 
                               timeout=30)
        
        if response.status_code == 201:
            try:
                post_data = response.json()
                st.success("✅ Post creation successful")
                st.write(f"Test post created with ID: {post_data.get('id')}")
                
                # Try to delete the test post
                delete_response = requests.delete(f"{wp_base}/wp-json/wp/v2/posts/{post_data.get('id')}", 
                                               headers=headers, timeout=15)
                if delete_response.status_code == 200:
                    st.success("✅ Test post deleted successfully")
                else:
                    st.warning(f"⚠️ Test post created but couldn't delete (ID: {post_data.get('id')})")
                
            except Exception as e:
                st.error(f"❌ Post creation response invalid: {str(e)}")
                return False
        else:
            st.error(f"❌ Post creation failed (HTTP {response.status_code})")
            if response.status_code == 500:
                st.code(f"Error details: {response.text[:500]}")
            try:
                error_data = response.json()
                st.write(f"Error message: {error_data.get('message', 'Unknown error')}")
            except:
                st.write(f"Raw response: {response.text[:200]}...")
            return False
    except Exception as e:
        st.error(f"❌ Post creation test error: {str(e)}")
        return False
    
    return True

# Handle detailed WordPress testing
if st.session_state.get("run_detailed_test"):
    st.header("🔍 WordPress Connection Diagnostic")
    
    if wp_config.get("base_url") and wp_config.get("username") and wp_config.get("password"):
        if test_wordpress_connection_detailed(wp_config):
            st.success("🎉 All tests passed! Your WordPress is ready for publishing.")
        else:
            st.error("❌ Some tests failed. Please check the errors above and your WordPress configuration.")
            
            # Common solutions for 500 errors
            st.subheader("💡 Solutions for HTTP 500 Errors")
            st.markdown("""
            **Application Password Issues:**
            - Go to WordPress Admin → Users → Your Profile
            - Scroll down to "Application Passwords"
            - Create a new application password
            - Copy the EXACT password with spaces (like: `abcd efgh ijkl mnop`)
            
            **Security Plugin Issues:**
            - Temporarily disable security plugins (WordFence, Sucuri, etc.)
            - Check if REST API is blocked in security settings
            - Look for "REST API" or "JSON API" settings in your security plugin
            
            **Server Configuration:**
            - Contact your hosting provider about REST API support
            - Check if mod_rewrite is enabled
            - Verify permalinks are working (Settings → Permalinks → Save)
            
            **Plugin Conflicts:**
            - Try deactivating plugins one by one to identify conflicts
            - Some caching or security plugins can interfere with authentication
            """)
    else:
        st.error("❌ WordPress configuration incomplete")
    
    # Clear the test flag
    st.session_state["run_detailed_test"] = False

# Handle 500 Error Fix Guide
if st.session_state.get("show_500_fix"):
    st.header("🛠️ WordPress 500 Error Fix Guide")
    
    st.markdown("""
    **You're getting HTTP 500 errors. Follow these steps in order:**
    
    ### Step 1: Generate New Application Password
    1. Go to your WordPress Admin Dashboard
    2. Navigate to **Users → Your Profile**
    3. Scroll down to **"Application Passwords"** section
    4. **Delete ALL existing application passwords**
    5. In the "Add New Application Password" field, enter: `Streamlit App`
    6. Click **"Add New Application Password"**
    7. **Copy the ENTIRE password including spaces** (e.g., `abcd efgh ijkl mnop`)
    8. Update the password in this app's configuration
    
    ### Step 2: Check Security Plugins
    """)
    
    security_plugins = [
        "WordFence Security",
        "Sucuri Security", 
        "iThemes Security",
        "All In One WP Security",
        "Jetpack Security"
    ]
    
    for plugin in security_plugins:
        st.write(f"- {plugin}")
    
    st.markdown("""
    **Temporarily deactivate these plugins** and test again.
    
    ### Step 3: Check REST API Settings
    Some plugins have specific REST API blocking settings:
    - Look for "REST API" or "JSON API" in plugin settings
    - Ensure REST API is not blocked for authenticated users
    
    ### Step 4: Plugin Conflict Test
    1. Deactivate **ALL plugins**
    2. Test the connection again
    3. If it works, reactivate plugins **one by one**
    4. Test after each activation to find the conflicting plugin
    
    ### Step 5: Permalink Settings
    1. Go to **Settings → Permalinks**
    2. Choose any permalink structure (not "Plain")
    3. Click **"Save Changes"**
    4. Test again
    
    ### Step 6: Contact Hosting Provider
    If none of the above work, contact your hosting provider and ask them to:
    - Check if REST API is blocked server-side
    - Enable mod_rewrite if it's disabled
    - Check error logs for specific PHP errors
    """)
    
    if st.button("✅ I've tried these steps"):
        st.session_state["show_500_fix"] = False
        st.rerun()

# Handle Auth Reset Guide  
if st.session_state.get("show_auth_reset"):
    st.header("🔄 Reset Authentication Guide")
    
    st.markdown("""
    **Follow these exact steps to reset your authentication:**
    
    ### WordPress Admin Steps:
    1. **Log into WordPress Admin** using your regular username/password
    2. Go to **Users → All Users**
    3. Click on your username to edit your profile
    4. Scroll down to **"Application Passwords"** section
    5. **Delete ALL existing application passwords** (click the X next to each)
    6. Create a new one:
       - Name: `StreamlitApp` 
       - Click **"Add New Application Password"**
    7. **IMPORTANT:** Copy the password EXACTLY as shown (with spaces)
    
    ### App Configuration Steps:
    1. In this app, go to **WordPress Configuration** in the sidebar
    2. Check **"Use Custom WordPress Settings"**
    3. Enter your details:
       - **Website URL:** `https://credsir.com` (no trailing slash)
       - **Username:** `nishitkumar` (case-sensitive)
       - **Application Password:** Paste the EXACT password with spaces
    4. Click **"Save Config"**
    5. Try the **"Quick Test"** button
    
    ### Common Mistakes to Avoid:
    - ❌ Don't remove spaces from the application password
    - ❌ Don't add /wp-admin to the website URL
    - ❌ Don't use your regular login password
    - ❌ Don't leave old application passwords active
    
    ### Test Format:
    Your application password should look like: `abcd efgh ijkl mnop` (with spaces)
    """)
    
    if st.button("✅ Authentication Reset Complete"):
        st.session_state["show_auth_reset"] = False
        st.rerun()
    
# Main App
st.title("📚 Enhanced SEO Content Automation")
st.markdown("Upload articles → Generate metadata → Create optimized images → Bulk publish to WordPress")

# Show current provider status
if current_api_key:
    st.success(f"✅ Connected to {ai_provider}")
else:
    st.error(f"❌ {ai_provider} API key not found in secrets")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Bulk Upload", 
    "📝 Metadata Generation", 
    "🖼️ Image Generation",
    "🔗 Internal Linking", 
    "🚀 WordPress Publish", 
    "📊 Export & Analytics"
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
    st.header("🖼️ Step 3: Image Generation")
    
    if st.session_state["uploaded_articles"] and st.session_state["article_metadata"]:
        st.subheader("Image Generation Options")
        
        # Single image generation
        article_files = [f for f in st.session_state["uploaded_articles"].keys() if f in st.session_state["article_metadata"]]
        selected_file = st.selectbox("Select article for image generation", article_files, key="img_select")
        
        if selected_file:
            article_data = st.session_state["uploaded_articles"][selected_file]
            metadata = st.session_state["article_metadata"][selected_file]
            
            st.write(f"**Article:** {article_data['title']}")
            st.write(f"**Primary Keyword:** {metadata.get('primary_keyword', 'N/A')}")
            
            # Image source selection
            image_source = st.radio(
                "Choose image source:",
                ["🤖 Generate with AI", "📁 Upload Existing Image"],
                horizontal=True
            )
            
            if image_source == "🤖 Generate with AI":
                if hf_client and current_api_key:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button("🤖 Generate AI Prompt", key="gen_prompt"):
                            with st.spinner("Creating optimized prompt..."):
                                optimized_prompt = generate_optimized_image_prompt(
                                    article_data['title'],
                                    article_data['content'],
                                    metadata.get('primary_keyword', ''),
                                    current_api_key,
                                    ai_provider
                                )
                                st.session_state[f"prompt_{selected_file}"] = optimized_prompt
                                st.rerun()
                    
                    with col2:
                        if st.button("📝 Use Default Prompt", key="default_prompt"):
                            default_prompt = f"Professional illustration for {metadata.get('primary_keyword', article_data['title'])}, clean modern design, educational content, high quality, minimalist style"
                            st.session_state[f"prompt_{selected_file}"] = default_prompt
                            st.rerun()
                    
                    # Editable prompt text area
                    image_prompt = st.text_area(
                        "✏️ **Edit Image Prompt:**",
                        value=st.session_state.get(f"prompt_{selected_file}", ""),
                        height=100,
                        help="Describe the image you want."
                    )
                    
                    # Generate image button
                    if st.button("🎨 Generate AI Image", type="primary"):
                        if image_prompt.strip():
                            with st.spinner("Generating image..."):
                                image_buffer = generate_ai_image(image_prompt.strip(), hf_client)
                                
                                if image_buffer:
                                    st.session_state["images"][selected_file] = {
                                        "buffer": image_buffer,
                                        "source": "ai_generated",
                                        "prompt": image_prompt.strip()
                                    }
                                    st.success("✅ Image generated successfully!")
                                    st.image(image_buffer, caption=f"Generated: {article_data['title']}")
                                else:
                                    st.error("❌ Image generation failed. Try a different prompt.")
                        else:
                            st.error("❌ Please enter a prompt before generating an image.")
                else:
                    st.error("❌ AI generation requires both HuggingFace token and API key.")
            
            else:  # Upload existing image
                uploaded_image = st.file_uploader(
                    "Upload an existing image",
                    type=['png', 'jpg', 'jpeg'],
                    help="Upload your own image for this article"
                )
                
                if uploaded_image:
                    # Convert uploaded file to buffer
                    image_buffer = BytesIO(uploaded_image.read())
                    st.session_state["images"][selected_file] = {
                        "buffer": image_buffer,
                        "source": "uploaded",
                        "filename": uploaded_image.name
                    }
                    st.success("✅ Image uploaded successfully!")
                    try:
                        image_buffer.seek(0)
                        st.image(image_buffer, caption=f"Uploaded: {uploaded_image.name}")
                    except Exception as e:
                        st.error(f"Error displaying uploaded image: {str(e)}")
            
            # Text overlay options (if image exists)
            if selected_file in st.session_state["images"]:
                st.subheader("🎨 Add Text Overlay")
                
                overlay_enabled = st.checkbox("Add text overlay to image", key=f"overlay_{selected_file}")
                
                if overlay_enabled:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        primary_overlay_text = st.text_input(
                            "Primary Text",
                            value=metadata.get('seo_title', article_data['title'])[:40] + "...",
                            help="Main headline text for overlay"
                        )
                    
                    with col2:
                        secondary_overlay_text = st.text_input(
                            "Secondary Text",
                            value=f"Complete Guide • {metadata.get('category', 'Article')}",
                            help="Subtitle or additional information"
                        )
                    
                    # Image dimensions for overlay
                    overlay_size = st.selectbox(
                        "Output Size",
                        ["Social Media (1200x675)", "Square (1080x1080)", "Story (1080x1920)", "Custom"],
                        help="Choose the output dimensions"
                    )
                    
                    if overlay_size == "Custom":
                        col1, col2 = st.columns(2)
                        with col1:
                            custom_width = st.number_input("Width", min_value=300, max_value=2000, value=1200)
                        with col2:
                            custom_height = st.number_input("Height", min_value=300, max_value=2000, value=675)
                        output_size = (custom_width, custom_height)
                    else:
                        size_map = {
                            "Social Media (1200x675)": (1200, 675),
                            "Square (1080x1080)": (1080, 1080),
                            "Story (1080x1920)": (1080, 1920)
                        }
                        output_size = size_map[overlay_size]
                    
                    if st.button("🎨 Apply Text Overlay", key=f"apply_overlay_{selected_file}"):
                        with st.spinner("Applying text overlay..."):
                            try:
                                # Load the base image
                                image_data = st.session_state["images"][selected_file]
                                image_data["buffer"].seek(0)
                                base_image = Image.open(image_data["buffer"])
                                
                                # Apply overlay
                                final_image = create_property_overlay(
                                    base_image,
                                    primary_overlay_text,
                                    secondary_overlay_text,
                                    output_size
                                )
                                
                                # Save the overlay image
                                overlay_buffer = BytesIO()
                                final_image.save(overlay_buffer, format='PNG')
                                overlay_buffer.seek(0)
                                
                                # Update the image with overlay
                                st.session_state["images"][selected_file] = {
                                    "buffer": overlay_buffer,
                                    "source": "overlay_applied",
                                    "original_source": image_data.get("source", "unknown"),
                                    "overlay_text": f"{primary_overlay_text} | {secondary_overlay_text}",
                                    "size": output_size
                                }
                                
                                st.success("✅ Text overlay applied!")
                                try:
                                    overlay_buffer.seek(0)
                                    st.image(overlay_buffer, caption=f"With Overlay: {article_data['title']}")
                                except Exception as e:
                                    st.error(f"Error displaying overlay image: {str(e)}")
                                
                            except Exception as e:
                                st.error(f"❌ Error applying overlay: {str(e)}")
        
        # Bulk image generation
        st.subheader("🚀 Bulk Image Generation")
        
        if hf_client and current_api_key:
            if st.button("🎨 Generate Images for All Articles"):
                progress_bar = st.progress(0)
                success_count = 0
                
                for i, file_name in enumerate(article_files):
                    article_data = st.session_state["uploaded_articles"][file_name]
                    metadata = st.session_state["article_metadata"][file_name]
                    
                    st.info(f"Generating image for: {article_data['title'][:50]}...")
                    
                    # Generate prompt
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
                        st.session_state["images"][file_name] = {
                            "buffer": image_buffer,
                            "source": "ai_generated_bulk",
                            "prompt": optimized_prompt
                        }
                        success_count += 1
                    
                    progress_bar.progress((i + 1) / len(article_files))
                    time.sleep(3)  # Rate limiting for image generation
                
                st.success(f"✅ Generated {success_count} images successfully!")
        
        # Display all generated images
        if st.session_state["images"]:
            st.subheader(f"🖼️ Generated Images ({len(st.session_state['images'])})")
            
            # Grid display
            cols = st.columns(3)
            for i, (file_name, image_data) in enumerate(st.session_state["images"].items()):
                with cols[i % 3]:
                    article_title = st.session_state["uploaded_articles"][file_name]['title']
                    try:
                        image_data["buffer"].seek(0)
                        st.image(image_data["buffer"], caption=f"{article_title[:25]}...")
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")
                        continue
                    
                    # Management buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"🗑️", key=f"del_{i}", help=f"Delete image"):
                            del st.session_state["images"][file_name]
                            st.rerun()
                    
                    with col_b:
                        try:
                            image_data["buffer"].seek(0)
                            st.download_button(
                                "⬇️",
                                image_data["buffer"].getvalue(),
                                file_name=f"{article_title[:30].replace(' ', '_')}.png",
                                mime="image/png",
                                key=f"down_{i}"
                            )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
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
        else:
            st.info("Click 'Fetch Posts from WordPress' to load existing posts for internal linking.")
    
    else:
        st.error("❌ WordPress not configured. Please configure in the sidebar.")

with tab5:
    st.header("🚀 Step 5: WordPress Publishing")
    
    if (st.session_state["uploaded_articles"] and 
        st.session_state["article_metadata"] and 
        wp_config.get("base_url") and 
        wp_config.get("username") and 
        wp_config.get("password")):
        
        # Publishing options
        col1, col2 = st.columns(2)
        
        with col1:
            publish_mode = st.radio("Publishing Mode", ["Draft", "Publish Immediately"])
            publish_now = publish_mode == "Publish Immediately"
        
        with col2:
            global_tags = st.text_input("Additional Tags (comma-separated)", "")
        
        # Single article publishing
        st.subheader("📝 Publish Single Article")
        
        ready_articles = [f for f in st.session_state["uploaded_articles"].keys() if f in st.session_state["article_metadata"]]
        selected_file = st.selectbox("Select article to publish", ready_articles, key="publish_select")
        
        if selected_file:
            article_data = st.session_state["uploaded_articles"][selected_file]
            metadata = st.session_state["article_metadata"][selected_file]
            
            # Preview
            with st.expander("📋 Publishing Preview", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Original Title:** {article_data['title']}")
                    st.write(f"**SEO Title:** {metadata.get('seo_title', 'Will use original title')}")
                    st.write(f"**Content Length:** {len(article_data['content'])} characters")
                    st.write(f"**Has Image:** {'✅ Yes' if selected_file in st.session_state['images'] else '❌ No'}")
                
                with col2:
                    st.write(f"**Primary Keyword:** {metadata.get('primary_keyword', 'None')}")
                    st.write(f"**Meta Description:** {metadata.get('meta_description', 'None')[:50]}...")
                    st.write(f"**Tags:** {len(metadata.get('tags', []))} tags")
                    st.write(f"**Category:** {metadata.get('category', 'None')}")
            
            # Publishing button
            if st.button("📤 Publish Selected Article", type="primary"):
                with st.spinner("Publishing to WordPress..."):
                    
                    # Prepare image buffer if available
                    image_buffer = None
                    if selected_file in st.session_state["images"]:
                        try:
                            image_data = st.session_state["images"][selected_file]
                            image_data["buffer"].seek(0)
                            image_buffer = image_data["buffer"]
                        except Exception as e:
                            st.warning(f"⚠️ Could not prepare image: {str(e)}")
                    
                    # Add global tags to metadata tags
                    final_metadata = metadata.copy()
                    if global_tags:
                        existing_tags = final_metadata.get('tags', [])
                        new_tags = [tag.strip() for tag in global_tags.split(',') if tag.strip()]
                        final_metadata['tags'] = existing_tags + new_tags
                    
                    # Publish
                    try:
                        result = publish_to_wordpress(
                            article_data['title'],
                            article_data['content'],
                            final_metadata,
                            image_buffer,
                            wp_config,
                            publish_now
                        )
                        
                        # Enhanced result display
                        if result["success"]:
                            st.success("✅ Published Successfully!")
                            st.success(f"🔗 **Article URL:** {result['url']}")
                            
                            # Log the publication
                            st.session_state["publish_log"].append({
                                "article": result.get('title', final_metadata.get('seo_title', article_data['title'])),
                                "url": result["url"],
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "status": "Published" if publish_now else "Draft"
                            })
                            
                            # Quick actions
                            st.link_button("🌐 View Article", result['url'])
                        
                        else:
                            st.error("❌ Publishing Failed")
                            st.error(f"**Error:** {result.get('error', 'Unknown error')}")
                            
                            # Add specific troubleshooting for 500 errors
                            if "500" in str(result.get('error', '')):
                                st.markdown("""
                                **Troubleshooting HTTP 500 Error:**
                                1. **Check Application Password**: Generate a new one in WordPress Admin → Users → Your Profile
                                2. **Security Plugins**: Temporarily disable security plugins (WordFence, Sucuri, etc.)
                                3. **REST API**: Ensure REST API is enabled in your WordPress settings
                                4. **Plugin Conflicts**: Try deactivating plugins to identify conflicts
                                5. **Server Logs**: Check your hosting provider's error logs for details
                                """)
                    
                    except Exception as e:
                        st.error(f"❌ Unexpected error during publishing: {str(e)}")
                        st.error("Please check your WordPress configuration and try again.")
        
        # Bulk publishing section
        st.subheader("🚀 Bulk Publishing")
        
        if ready_articles:
            st.info(f"📊 Ready to publish: **{len(ready_articles)}** articles")
            
            # Bulk publishing options
            col1, col2 = st.columns(2)
            
            with col1:
                bulk_delay = st.selectbox("Delay Between Posts", ["2 seconds", "5 seconds", "10 seconds"], index=1)
                delay_seconds = int(bulk_delay.split()[0])
            
            with col2:
                skip_failed = st.checkbox("Skip Failed Articles", value=True, help="Continue publishing even if some articles fail")
            
            # Bulk publishing button
            if st.button("🚀 Publish All Articles", type="primary"):
                progress_bar = st.progress(0)
                status_container = st.container()
                
                success_count = 0
                failed_count = 0
                
                for i, file_name in enumerate(ready_articles):
                    article_data = st.session_state["uploaded_articles"][file_name]
                    metadata = st.session_state["article_metadata"][file_name]
                    
                    current_title = metadata.get('seo_title', article_data['title'])
                    
                    with status_container:
                        st.info(f"📤 Publishing {i+1}/{len(ready_articles)}: {current_title[:50]}...")
                    
                    # Prepare image buffer
                    image_buffer = None
                    if file_name in st.session_state["images"]:
                        try:
                            image_data = st.session_state["images"][file_name]
                            image_data["buffer"].seek(0)
                            image_buffer = image_data["buffer"]
                        except:
                            pass  # Continue without image
                    
                    # Add global tags to metadata tags
                    final_metadata = metadata.copy()
                    if global_tags:
                        existing_tags = final_metadata.get('tags', [])
                        new_tags = [tag.strip() for tag in global_tags.split(',') if tag.strip()]
                        final_metadata['tags'] = existing_tags + new_tags
                    
                    # Publish article
                    try:
                        result = publish_to_wordpress(
                            article_data['title'],
                            article_data['content'],
                            final_metadata,
                            image_buffer,
                            wp_config,
                            publish_now
                        )
                        
                        if result["success"]:
                            success_count += 1
                            
                            with status_container:
                                st.success(f"✅ Published: {current_title[:40]}... - {result['url']}")
                            
                            # Log publication
                            st.session_state["publish_log"].append({
                                "article": result.get("title", current_title),
                                "url": result["url"],
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "status": "Published" if publish_now else "Draft"
                            })
                        
                        else:
                            failed_count += 1
                            
                            with status_container:
                                st.error(f"❌ Failed: {current_title[:40]}... - {result.get('error', 'Unknown error')}")
                            
                            # Stop if not skipping failed articles
                            if not skip_failed:
                                st.error(f"⏹️ Bulk publishing stopped due to failure. {success_count} articles published successfully.")
                                break
                    
                    except Exception as e:
                        failed_count += 1
                        with status_container:
                            st.error(f"❌ Exception: {current_title[:40]}... - {str(e)}")
                        
                        if not skip_failed:
                            st.error(f"⏹️ Bulk publishing stopped due to exception. {success_count} articles published successfully.")
                            break
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(ready_articles))
                    
                    # Rate limiting delay
                    if i < len(ready_articles) - 1:  # Don't delay after the last article
                        time.sleep(delay_seconds)
                
                # Final summary
                with status_container:
                    st.success(f"🎉 Bulk Publishing Completed!")
                    st.info(f"✅ **{success_count}** articles published successfully")
                    if failed_count > 0:
                        st.warning(f"❌ **{failed_count}** articles failed")
        
        else:
            st.warning("⚠️ No articles ready for publishing. Complete Steps 1-2 first.")
    
    else:
        missing = []
        if not st.session_state["uploaded_articles"]:
            missing.append("uploaded articles")
        if not st.session_state["article_metadata"]:
            missing.append("article metadata")
        if not wp_config.get("base_url"):
            missing.append("WordPress URL")
        if not wp_config.get("username"):
            missing.append("WordPress username")
        if not wp_config.get("password"):
            missing.append("WordPress password")
        
        st.error(f"❌ **Missing:** {', '.join(missing)}")

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
            
            col1, col2, col3 = st.columns(3)
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
            
            with col3:
                # Image export (if available)
                if selected_file in st.session_state["images"]:
                    try:
                        image_data = st.session_state["images"][selected_file]
                        image_data["buffer"].seek(0)
                        st.download_button(
                            "⬇️ Download Image",
                            image_data["buffer"].getvalue(),
                            file_name=f"{article_data['title'][:30].replace(' ', '_')}.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.button("⬇️ Image Error", disabled=True, help=f"Error: {str(e)}")
                else:
                    st.button("⬇️ No Image", disabled=True)
        
        # Bulk export
        if st.button("📦 Create Complete Export Package"):
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add articles
                for file_name, article_data in st.session_state["uploaded_articles"].items():
                    safe_title = article_data['title'][:30].replace(' ', '_').replace('/', '_').replace('\\', '_')
                    
                    # HTML version
                    zip_file.writestr(f"articles/{safe_title}.html", article_data['content'])
                    
                    # Text version
                    clean_content = re.sub(r'<[^>]+>', '', article_data['content'])
                    zip_file.writestr(f"articles_text/{safe_title}.txt", clean_content)
                
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
                for file_name, image_data in st.session_state["images"].items():
                    try:
                        article_title = st.session_state["uploaded_articles"][file_name]['title']
                        safe_title = article_title[:30].replace(' ', '_').replace('/', '_').replace('\\', '_')
                        
                        image_data["buffer"].seek(0)
                        zip_file.writestr(f"images/{safe_title}.png", image_data["buffer"].read())
                    except Exception as e:
                        st.warning(f"Skipped corrupted image for {file_name}: {str(e)}")
                        continue
            
            zip_buffer.seek(0)
            
            st.download_button(
                "⬇️ Download Complete Package (ZIP)",
                zip_buffer,
                file_name=f"seo_content_package_{time.strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip"
            )
    
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📚 Enhanced SEO Content Automation Pipeline | Built with Streamlit</p>
    <p>Upload → Analyze → Optimize → Link → Publish | Complete workflow with image management</p>
</div>
""", unsafe_allow_html=True)
