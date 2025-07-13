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
            
            # Show current image with management options
            if selected_file in st.session_state["images"]:
                st.subheader("🖼️ Current Image")
                
                image_data = st.session_state["images"][selected_file]
                
                # Reset buffer position before displaying
                try:
                    image_data["buffer"].seek(0)
                    st.image(image_data["buffer"], caption=f"Current image for: {article_data['title']}")
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
                    # Remove corrupted image data
                    del st.session_state["images"][selected_file]
                    st.warning("Corrupted image data removed. Please regenerate the image.")
                    st.rerun()
                
                # Image management buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🗑️ Remove Image", key=f"remove_img_{selected_file}"):
                        del st.session_state["images"][selected_file]
                        st.success("✅ Image removed!")
                        st.rerun()
                
                with col2:
                    # Download button
                    try:
                        image_data["buffer"].seek(0)
                        clean_title = article_data['title'][:30].replace(' ', '_')
                        st.download_button(
                            "⬇️ Download Image",
                            image_data["buffer"].getvalue(),
                            file_name=f"{clean_title}.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Error preparing download: {str(e)}")
                
                with col3:
                    # Show image info
                    if st.button("ℹ️ Image Info"):
                        info_text = f"""
                        **Source:** {image_data.get('source', 'Unknown')}
                        **Size:** {image_data.get('size', 'Original')}
                        **Overlay:** {'Yes' if 'overlay_text' in image_data else 'No'}
                        """
                        st.info(info_text)
        
        # Bulk image generation
        st.subheader("🚀 Bulk Image Generation")
        
        bulk_image_source = st.radio(
            "Bulk generation source:",
            ["🤖 AI Generated Only", "📁 Skip articles with existing images"],
            horizontal=True
        )
        
        if bulk_image_source == "🤖 AI Generated Only" and hf_client and current_api_key:
            bulk_prompt_mode = st.radio(
                "Bulk Generation Mode:",
                ["🤖 AI-Generated Prompts", "📝 Custom Template", "⚡ Simple Default"]
            )
            
            if bulk_prompt_mode == "📝 Custom Template":
                bulk_template = st.text_area(
                    "Custom Prompt Template (use {keyword} and {title} as placeholders):",
                    value="Professional illustration about {keyword}, clean modern design, {title} concept, minimalist style",
                    help="Use {keyword} for primary keyword and {title} for article title"
                )
            
            if st.button("🎨 Generate Images for All Articles"):
                progress_bar = st.progress(0)
                success_count = 0
                
                for i, file_name in enumerate(article_files):
                    # Skip if image already exists and mode is set to skip
                    if bulk_image_source == "📁 Skip articles with existing images" and file_name in st.session_state["images"]:
                        continue
                    
                    article_data = st.session_state["uploaded_articles"][file_name]
                    metadata = st.session_state["article_metadata"][file_name]
                    
                    st.info(f"Generating image for: {article_data['title'][:50]}...")
                    
                    # Generate prompt based on mode
                    if bulk_prompt_mode == "🤖 AI-Generated Prompts":
                        optimized_prompt = generate_optimized_image_prompt(
                            article_data['title'],
                            article_data['content'],
                            metadata.get('primary_keyword', ''),
                            current_api_key,
                            ai_provider
                        )
                    elif bulk_prompt_mode == "📝 Custom Template":
                        keyword_val = metadata.get('primary_keyword', article_data['title'])
                        title_val = article_data['title']
                        optimized_prompt = bulk_template.format(
                            keyword=keyword_val,
                            title=title_val
                        )
                    else:  # Simple Default
                        primary_kw = metadata.get('primary_keyword', article_data['title'])
                        optimized_prompt = f"Professional illustration about {primary_kw}, clean modern design, educational content, minimalist style"
                    
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
                    
                    # Small management buttons for each image
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"🔄", key=f"regen_{i}", help=f"Regenerate image"):
                            # Show regeneration options in expander
                            with st.expander(f"Regenerate {article_title[:20]}..."):
                                st.write("Choose regeneration method:")
                                if st.button("🤖 AI Prompt", key=f"ai_regen_{i}"):
                                    if current_api_key and hf_client:
                                        metadata = st.session_state["article_metadata"][file_name]
                                        new_prompt = generate_optimized_image_prompt(
                                            article_title,
                                            st.session_state["uploaded_articles"][file_name]['content'],
                                            metadata.get('primary_keyword', ''),
                                            current_api_key,
                                            ai_provider
                                        )
                                        new_image = generate_ai_image(new_prompt, hf_client)
                                        if new_image:
                                            st.session_state["images"][file_name] = {
                                                "buffer": new_image,
                                                "source": "ai_regenerated",
                                                "prompt": new_prompt
                                            }
                                            st.rerun()
                    
                    with col_b:
                        if st.button(f"🗑️", key=f"del_{i}", help=f"Delete image"):
                            del st.session_state["images"][file_name]
                            st.rerun()
    
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
        
        st.subheader("📤 Streamlined Publishing")
        st.info("🚀 **Simple & Fast**: Articles will be published as HTML with metadata. Tags and images are optional - the system won't halt if they fail.")
        
        publish_mode = st.radio("Publishing Mode", ["Draft", "Publish Immediately"])
        publish_now = publish_mode == "Publish Immediately"
        
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
                st.write(f"**Original Title:** {article_data['title'][:50]}...")
                seo_title = metadata.get('seo_title', 'Using original title')
                st.write(f"**SEO Title:** {seo_title[:50]}...")
                has_image_text = '✅' if selected_file in st.session_state['images'] else '❌'
                st.write(f"**Has Image:** {has_image_text}")
            
            with col2:
                st.write(f"**Primary Keyword:** {metadata.get('primary_keyword', 'None')}")
                st.write(f"**Category:** {metadata.get('category', 'None')}")
                st.write(f"**Status:** {publish_mode}")
            
            if st.button("📤 Publish Selected Article"):
                with st.spinner("Publishing to WordPress..."):
                    # Get image buffer if available
                    image_buffer = None
                    if selected_file in st.session_state["images"]:
                        try:
                            image_data = st.session_state["images"][selected_file]
                            image_data["buffer"].seek(0)
                            image_buffer = image_data["buffer"]
                        except:
                            pass  # Continue without image if there's an error
                    
                    # Publish using streamlined function
                    result = publish_to_wordpress_streamlined(
                        article_data['title'],
                        article_data['content'],
                        metadata,
                        image_buffer,
                        wp_config,
                        publish_now
                    )
                    
                    if result["success"]:
                        st.markdown('<div class="success-box">✅ Published successfully!</div>', unsafe_allow_html=True)
                        st.success(f"🔗 **URL:** {result['url']}")
                        has_img_text = '✅ Uploaded' if result.get('has_image') else '❌ Not uploaded'
                        st.info(f"📸 **Image:** {has_img_text}")
                        
                        # Log the publication
                        st.session_state["publish_log"].append({
                            "article": result["title"],
                            "url": result["url"],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "Published" if publish_now else "Draft",
                            "has_image": "Yes" if result.get('has_image') else "No",
                            "has_metadata": "Yes"
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
                    
                    seo_title = metadata.get('seo_title', article_data['title'])
                    st.write(f"**{seo_title}** - Image: {has_image}")
            
            if st.button("🚀 Publish All Articles"):
                progress_bar = st.progress(0)
                success_count = 0
                
                for i, file_name in enumerate(ready_articles):
                    article_data = st.session_state["uploaded_articles"][file_name]
                    metadata = st.session_state["article_metadata"][file_name]
                    
                    display_title = metadata.get('seo_title', article_data['title'])
                    st.info(f"Publishing: {display_title[:50]}...")
                    
                    # Get image if available
                    image_buffer = None
                    if file_name in st.session_state["images"]:
                        try:
                            image_data = st.session_state["images"][file_name]
                            image_data["buffer"].seek(0)
                            image_buffer = image_data["buffer"]
                        except:
                            pass  # Continue without image if there's an error
                    
                    # Publish
                    result = publish_to_wordpress_streamlined(
                        article_data['title'],
                        article_data['content'],
                        metadata,
                        image_buffer,
                        wp_config,
                        publish_now
                    )
                    
                    if result["success"]:
                        success_count += 1
                        st.success(f"✅ Published: {display_title[:30]}...")
                        
                        # Log publication
                        st.session_state["publish_log"].append({
                            "article": display_title,
                            "url": result["url"],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "Published" if publish_now else "Draft",
                            "has_image": "Yes" if result.get('has_image') else "No"
                        })
                    else:
                        st.error(f"❌ Failed: {display_title[:30]}...")
                    
                    progress_bar.progress((i + 1) / len(ready_articles))
                    time.sleep(2)  # Rate limiting
                
                success_msg = f'🎉 Bulk publishing completed! {success_count}/{len(ready_articles)} articles published successfully.'
                st.markdown(f'<div class="success-box">{success_msg}</div>', unsafe_allow_html=True)
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
        
        # Enhanced Analytics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Published", len(log_df))
        with col2:
            published_count = len(log_df[log_df["status"] == "Published"])
            st.metric("Live Articles", published_count)
        with col3:
            draft_count = len(log_df[log_df["status"] == "Draft"])
            st.metric("Drafts", draft_count)
        with col4:
            with_images = len(log_df[log_df.get("has_image", "No") == "Yes"])
            st.metric("With Images", with_images)
        
        # Export log
        log_csv = log_df.to_csv(index=False)
        current_date = time.strftime('%Y%m%d')
        st.download_button(
            "⬇️ Download Publication Log",
            log_csv,
            file_name=f"publication_log_{current_date}.csv",
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
                clean_title = article_data['title'][:30].replace(' ', '_')
                st.download_button(
                    "⬇️ Download HTML",
                    article_data['content'],
                    file_name=f"{clean_title}.html",
                    mime="text/html"
                )
            
            with col2:
                # Text export
                clean_content = re.sub(r'<[^>]+>', '', article_data['content'])
                st.download_button(
                    "⬇️ Download Text",
                    clean_content,
                    file_name=f"{clean_title}.txt",
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
                            file_name=f"{clean_title}.png",
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
            
            current_datetime = time.strftime('%Y%m%d_%H%M')
            st.download_button(
                "⬇️ Download Complete Package (ZIP)",
                zip_buffer,
                file_name=f"seo_content_package_{current_datetime}.zip",
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
        2. **Tag Management**: Test WordPress connection to fetch existing tags before publishing
        3. **Image Generation**: Use AI-optimized prompts for better results, add text overlays for marketing appeal
        4. **Image Management**: Upload existing images or generate new ones, delete unwanted images easily
        5. **Text Overlays**: Use MagicBricks-style overlays for professional marketing images
        6. **Internal Linking**: Fetch existing posts before running bulk operations
        7. **WordPress**: Test connection before bulk publishing, tags will be created if they don't exist
        8. **Backup**: Always export your work before major operations
        
        **New Features:**
        - **Enhanced Tag Handling**: Automatically fetches existing WordPress tags and creates new ones as needed
        - **Image Upload Option**: Upload existing images instead of generating new ones
        - **Text Overlays**: Add professional marketing text overlays to images with custom sizing
        - **Image Management**: Easy delete/regenerate options for all images
        - **Improved Error Handling**: Better error messages and fallback options
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📚 Enhanced SEO Content Automation Pipeline | Built with Streamlit</p>
    <p>Upload → Analyze → Optimize → Link → Publish | Complete workflow with image management and text overlays</p>
</div>
""", unsafe_allow_html=True)def find_internal_linking_opportunities(article_content, existing_posts, api_key, provider):
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
    if metadata.get('meta_description'):
        meta_comment = f"<!-- Meta Description: {metadata['meta_description']} -->\n"
        html_content = meta_comment + html_content
    
    # Add primary keyword as hidden comment
    if metadata.get('primary_keyword'):
        keyword_comment = f"<!-- Primary Keyword: {metadata['primary_keyword']} -->\n"
        html_content = keyword_comment + html_content
    
    # Ensure proper paragraph structure
    if '<p>' not in html_content and '<div>' not in html_content:
        # Wrap content in paragraphs
        paragraphs = html_content.split('\n\n')
        html_content = ''.join([f"<p>{p.strip()}</p>\n" for p in paragraphs if p.strip()])
    
    return html_content

def publish_to_wordpress_streamlined(title, content, metadata, image_buffer, wp_config, publish_now=True):
    """Streamlined WordPress publishing - never halts, always publishes"""
    
    auth_str = f"{wp_config['username']}:{wp_config['password']}"
    auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
    headers = {
        "Authorization": f"Basic {auth_token}",
        "Content-Type": "application/json"
    }
    
    # Prepare HTML content with metadata
    html_content = prepare_article_html(content, metadata)
    
    # Use SEO title if available, otherwise use original title
    final_title = metadata.get('seo_title', title)
    
    # Try to upload image (but don't halt if it fails)
    img_id = None
    if image_buffer:
        try:
            image_buffer.seek(0)
            img_data = image_buffer.read()
            img_headers = headers.copy()
            img_headers.update({
                "Content-Disposition": f"attachment; filename={final_title.replace(' ', '_')}.jpg",
                "Content-Type": "image/jpeg"
            })
            media_url = f"{wp_config['base_url']}/wp-json/wp/v2/media"
            img_resp = requests.post(media_url, headers=img_headers, data=img_data, timeout=15)
            
            if img_resp.status_code == 201:
                img_id = img_resp.json()["id"]
            # If image upload fails, just continue without it
        except:
            # Silently continue if image upload fails
            pass
    
    # Create post data - minimal and robust
    post_data = {
        "title": final_title,
        "content": html_content,
        "status": "publish" if publish_now else "draft"
    }
    
    # Add image only if upload succeeded
    if img_id:
        post_data["featured_media"] = img_id
    
    # Add excerpt from meta description if available
    if metadata.get('meta_description'):
        post_data["excerpt"] = metadata['meta_description']
    
    # Publish the article
    try:
        post_url = f"{wp_config['base_url']}/wp-json/wp/v2/posts"
        post_resp = requests.post(post_url, headers=headers, json=post_data, timeout=20)
        
        if post_resp.status_code == 201:
            post_data_response = post_resp.json()
            article_url = post_data_response["link"]
            
            return {
                "success": True, 
                "url": article_url,
                "has_image": bool(img_id),
                "title": final_title
            }
        else:
            return {
                "success": False, 
                "error": f"Publishing failed: HTTP {post_resp.status_code}",
                "title": final_title
            }
    
    except Exception as e:
        return {
            "success": False, 
            "error": f"Publishing error: {str(e)}",
            "title": final_title
        }

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
    
    # Test connection and fetch tags
    if st.sidebar.button("🔍 Test WordPress Connection"):
        try:
            auth_str = f"{wp_config['username']}:{wp_config['password']}"
            auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
            headers = {"Authorization": f"Basic {auth_token}"}
            
            # Test basic API access
            test_url = f"{wp_config['base_url']}/wp-json/wp/v2/posts?per_page=1"
            response = requests.get(test_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                st.sidebar.success("✅ WordPress connection successful!")
                
                # Fetch existing tags
                with st.sidebar:
                    with st.spinner("Fetching existing tags..."):
                        existing_tags = get_wordpress_tags(wp_config)
                        st.session_state["existing_tags"] = existing_tags
                        
                        if existing_tags:
                            st.sidebar.info(f"📋 Fetched {len(existing_tags)} existing tags")
                        else:
                            st.sidebar.info("📋 No existing tags found or unable to fetch tags")
            
            elif response.status_code == 401:
                st.sidebar.error("❌ Authentication failed: Invalid username or password")
            elif response.status_code == 403:
                st.sidebar.error("❌ Access forbidden: Check user permissions")
            elif response.status_code == 404:
                st.sidebar.error("❌ WordPress REST API not found. Check URL and ensure REST API is enabled")
            else:
                response_text = response.text[:100] + "..." if len(response.text) > 100 else response.text
                st.sidebar.error(f"❌ Connection failed: HTTP {response.status_code}")
                st.sidebar.error(f"Response: {response_text}")
        
        except requests.exceptions.Timeout:
            st.sidebar.error("❌ Connection timeout. Please check your WordPress URL.")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"❌ Network error: {str(e)}")
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

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .image-option-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .tag-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
                            
                            # Display metadata with tag management
                            st.json(metadata)
                            
                            # Tag management section
                            st.markdown('<div class="tag-box">', unsafe_allow_html=True)
                            st.write("**🏷️ Tag Management:**")
                            
                            # Show existing tags for reference
                            if st.session_state.get("existing_tags"):
                                existing_tag_names = [tag["name"] for tag in st.session_state["existing_tags"]]
                                st.write(f"**Existing WordPress Tags:** {', '.join(existing_tag_names[:10])}{'...' if len(existing_tag_names) > 10 else ''}")
                            
                            # Allow editing generated tags
                            current_tags = metadata.get("tags", [])
                            edited_tags = st.text_input(
                                "Edit Tags (comma-separated)",
                                value=", ".join(current_tags),
                                help="Modify the generated tags or add custom ones"
                            )
                            
                            if st.button("💾 Update Tags"):
                                new_tags = [tag.strip() for tag in edited_tags.split(',') if tag.strip()]
                                st.session_state["article_metadata"][selected_file]["tags"] = new_tags
                                st.success("✅ Tags updated!")
                                st.rerun()
                            
                            st.markdown('</div>', unsafe_allow_html=True)
        
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
    st.header("🖼️ Step 3: Enhanced Image Generation")
    
    if st.session_state["uploaded_articles"] and st.session_state["article_metadata"]:
        st.subheader("Image Generation Options")
        
        # Single image generation with multiple options
        article_files = [f for f in st.session_state["uploaded_articles"].keys() if f in st.session_state["article_metadata"]]
        selected_file = st.selectbox("Select article for image generation", article_files, key="img_select")
        
        if selected_file:
            article_data = st.session_state["uploaded_articles"][selected_file]
            metadata = st.session_state["article_metadata"][selected_file]
            
            st.markdown('<div class="image-option-box">', unsafe_allow_html=True)
            st.write(f"**Article:** {article_data['title']}")
            st.write(f"**Primary Keyword:** {metadata.get('primary_keyword', 'N/A')}")
            st.write(f"**Category:** {metadata.get('category', 'N/A')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image source selection
            image_source = st.radio(
                "Choose image source:",
                ["🤖 Generate with AI", "📁 Upload Existing Image"],
                horizontal=True
            )
            
            # Initialize session state for prompts
            if f"prompt_{selected_file}" not in st.session_state:
                st.session_state[f"prompt_{selected_file}"] = ""
            
            if image_source == "🤖 Generate with AI":
                if hf_client and current_api_key:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Button to generate AI prompt
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
                        # Button to use default prompt
                        if st.button("📝 Use Default Prompt", key="default_prompt"):
                            primary_kw = metadata.get('primary_keyword', article_data['title'])
                            default_prompt = f"Professional illustration for {primary_kw}, clean modern design, educational content, high quality, minimalist style"
                            st.session_state[f"prompt_{selected_file}"] = default_prompt
                            st.rerun()
                    
                    # Editable prompt text area
                    image_prompt = st.text_area(
                        "✏️ **Edit Image Prompt** (You can modify this before generating):",
                        value=st.session_state[f"prompt_{selected_file}"],
                        height=100,
                        help="Describe the image you want. For best results with Stable Diffusion: use simple, clear descriptions, avoid complex scenes, focus on single subjects, use professional style keywords."
                    )
                    
                    # Update the prompt in session state when text area changes
                    if image_prompt != st.session_state[f"prompt_{selected_file}"]:
                        st.session_state[f"prompt_{selected_file}"] = image_prompt
                    
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
                                    # Display the generated image
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
                        # Remove corrupted upload
                        if selected_file in st.session_state["images"]:
                            del st.session_state["images"][selected_file]
            
            # Text overlay options (if image exists)
            if selected_file in st.session_state["images"]:
                st.subheader("🎨 Add Text Overlay")
                
                # Overlay text inputs
                overlay_enabled = st.checkbox("Add text overlay to image", key=f"overlay_{selected_file}")
                
                if overlay_enabled:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        seo_title = metadata.get('seo_title', article_data['title'])
                        primary_overlay_text = st.text_input(
                            "Primary Text",
                            value=seo_title[:40] + "..." if len(seo_title) > 40 else seo_title,
                            help="Main headline text for overlay"
                        )
                    
                    with col2:
                        category = metadata.get('category', 'Article')
                        secondary_overlay_text = st.text_input(
                            "Secondary Text",
                            value=f"Complete Guide • {category}",
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
    """Fetch existing WordPress tags with enhanced error handling"""
    try:
        auth_str = f"{wp_config['username']}:{wp_config['password']}"
        auth_token = base64.b64encode(auth_str.encode()).decode("utf-8")
        headers = {"Authorization": f"Basic {auth_token}"}
        
        url = f"{wp_config['base_url']}/wp-json/wp/v2/tags?per_page=100"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            try:
                tags_data = response.json()
                return [{"id": tag["id"], "name": tag["name"], "slug": tag["slug"]} for tag in tags_data]
            except (json.JSONDecodeError, KeyError) as e:
                st.warning(f"Error parsing tags response: {str(e)}")
                return []
        
        elif response.status_code == 401:
            st.error("❌ WordPress authentication failed. Please check your credentials.")
            return []
        elif response.status_code == 403:
            st.error("❌ WordPress access forbidden. Please check your user permissions.")
            return []
        elif response.status_code == 404:
            st.warning("❌ WordPress REST API endpoint not found. Please ensure WordPress is properly configured.")
            return []
        else:
            response_text = response.text[:200] + "..." if len(response.text) > 200 else response.text
            st.warning(f"Failed to fetch tags: HTTP {response.status_code}")
            st.warning(f"Response preview: {response_text}")
            return []
            
    except requests.exceptions.Timeout:
        st.warning("Timeout fetching WordPress tags. Please check your connection.")
        return []
    except requests.exceptions.RequestException as e:
        st.warning(f"Network error fetching tags: {str(e)}")
        return []
    except Exception as e:
        st.warning(f"Unexpected error fetching tags: {str(e)}")
        return []

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
        marketing_fonts = [
            "arialbd.ttf", "calibrib.ttf", "arial.ttf", "calibri.ttf",
            "/System/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/Helvetica-Bold.ttf",
            "/System/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Helvetica.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]
        
        for font_path in marketing_fonts:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        
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
        
        # Calculate total height needed
        primary_bbox = draw.textbbox((0, 0), primary_lines[0], font=primary_font)
        line_height = primary_bbox[3] - primary_bbox[1]
        total_primary_height = len(primary_lines) * line_height + (len(primary_lines) - 1) * 5
        
        # Position primary text
        current_y = text_start_y
        for line in primary_lines:
            bbox = draw.textbbox((0, 0), line, font=primary_font)
            line_width = bbox[2] - bbox[0]
            line_x = (width - line_width) // 2
            
            draw_marketing_text(line, (line_x, current_y), primary_font)
            current_y += line_height + 5
        
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
                
                secondary_bbox = draw.textbbox((0, 0), line, font=secondary_font)
                current_y += (secondary_bbox[3] - secondary_bbox[1]) + 5
    
    # Composite the overlay onto the image
    final_img = Image.alpha_composite(img.convert("RGBA"), overlay)
    
    # Apply subtle blur to the gradient area for smoother dissolution
    mask = Image.new("L", output_size, 0)
    mask_draw = ImageDraw.Draw(mask)
    
    # Create mask for gradient area only
    for y in range(gradient_height):
        alpha = int(255 * (y / gradient_height))
        mask_draw.rectangle([gradient_start_x, gradient_start_y + y, 
                           gradient_start_x + gradient_width, gradient_start_y + y + 1], 
                           fill=alpha)
    
    # Apply subtle blur to gradient area
    blur_radius = max(0.5, min(2.0, width / 2400))  # Responsive blur
    blurred = final_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    final_img = Image.composite(blurred, final_img, mask)
    
    # Enhance for marketing appeal
    enhancer = ImageEnhance.Contrast(final_img)
    final_img = enhancer.enhance(1.08)
    
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
