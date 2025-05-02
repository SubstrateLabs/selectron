from datetime import datetime, timezone

from selectron.lib.extract_metadata import HtmlMetadata, extract_metadata


class TestHtmlMetadata:
    def test_empty_html(self):
        """Test extraction from empty HTML"""
        html = ""
        metadata = extract_metadata(html)
        assert isinstance(metadata, HtmlMetadata)
        assert metadata.title is None
        assert metadata.description is None
        assert metadata.author is None
        assert metadata.og_image is None
        assert metadata.favicon is None
        assert metadata.keywords is None
        assert metadata.published_at is None

    def test_minimal_html(self):
        """Test extraction from minimal HTML with just a title"""
        html = "<html><head><title>Minimal Title</title></head><body></body></html>"
        metadata = extract_metadata(html)
        assert metadata.title == "Minimal Title"
        assert metadata.description is None
        assert metadata.author is None
        assert metadata.og_image is None
        assert metadata.favicon is None
        assert metadata.keywords is None
        assert metadata.published_at is None

    def test_complete_html(self):
        """Test extraction from HTML with all metadata fields"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Page Title</title>
            <meta name="description" content="Page description">
            <meta name="author" content="John Doe">
            <meta property="og:title" content="OG Title">
            <meta property="og:description" content="OG description">
            <meta property="og:image" content="https://example.com/image.jpg">
            <link rel="icon" href="/favicon.ico">
            <meta name="keywords" content="test, metadata, extraction">
            <meta property="article:published_time" content="2023-05-01T12:00:00+00:00">
        </head>
        <body>Content</body>
        </html>
        """
        metadata = extract_metadata(html, url="https://example.com")

        # OpenGraph title should take precedence over regular title
        assert metadata.title == "OG Title"
        # OpenGraph description should take precedence over meta description
        assert metadata.description == "OG description"
        assert metadata.author == "John Doe"
        assert metadata.og_image == "https://example.com/image.jpg"
        assert metadata.favicon == "/favicon.ico"
        assert metadata.url == "https://example.com"
        assert metadata.keywords == ["test", "metadata", "extraction"]
        assert metadata.published_at == datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_precedence_order(self):
        """Test that extraction follows the correct precedence order"""
        html = """
        <html>
        <head>
            <title>Regular Title</title>
            <meta name="description" content="Regular description">
            <meta property="og:title" content="OG Title">
            <meta property="og:description" content="OG description">
            <meta property="og:image" content="https://example.com/image.jpg">
        </head>
        <body></body>
        </html>
        """
        metadata = extract_metadata(html)

        # OG title should override regular title
        assert metadata.title == "OG Title"
        # OG description should override regular description
        assert metadata.description == "OG description"

    def test_fallback_to_regular_tags(self):
        """Test fallback to regular tags when OG tags are missing"""
        html = """
        <html>
        <head>
            <title>Regular Title</title>
            <meta name="description" content="Regular description">
        </head>
        <body></body>
        </html>
        """
        metadata = extract_metadata(html)

        assert metadata.title == "Regular Title"
        assert metadata.description == "Regular description"

    def test_multiple_favicon_formats(self):
        """Test extraction of different favicon formats"""
        # Test different favicon rel attributes
        html_icon = '<html><head><link rel="icon" href="/icon.ico"></head></html>'
        metadata_icon = extract_metadata(html_icon)
        assert metadata_icon.favicon == "/icon.ico"

        html_shortcut = '<html><head><link rel="shortcut icon" href="/shortcut.ico"></head></html>'
        metadata_shortcut = extract_metadata(html_shortcut)
        assert metadata_shortcut.favicon == "/shortcut.ico"

        html_apple = '<html><head><link rel="apple-touch-icon" href="/apple.png"></head></html>'
        metadata_apple = extract_metadata(html_apple)
        assert metadata_apple.favicon == "/apple.png"

    def test_whitespace_handling(self):
        """Test whitespace is properly stripped from extracted values"""
        html = """
        <html>
        <head>
            <title>  Title with spaces  </title>
            <meta name="description" content="  Description with spaces  ">
            <meta property="og:image" content="  https://example.com/image.jpg  ">
            <link rel="icon" href="  /favicon.ico  ">
        </head>
        <body></body>
        </html>
        """
        metadata = extract_metadata(html)

        assert metadata.title == "Title with spaces"
        assert metadata.description == "Description with spaces"
        assert metadata.og_image == "https://example.com/image.jpg"
        assert metadata.favicon == "/favicon.ico"

    def test_malformed_html(self):
        """Test extraction from malformed HTML"""
        html = """
        <html>
        <head>
            <title>Malformed
            <meta name="description" content="Description">
        </head>
        <body>
        """
        metadata = extract_metadata(html)

        # BeautifulSoup should handle the malformed HTML gracefully
        # We expect it to find the description but may not be able to extract the title
        # due to the unclosed tag
        assert metadata.description == "Description"
        # We're not asserting anything about the title as its behavior may vary

    def test_special_characters(self):
        """Test extraction with special characters"""
        html = """
        <html>
        <head>
            <title>Title with &amp; and &lt; entities</title>
            <meta name="description" content="Description with © symbol">
        </head>
        <body></body>
        </html>
        """
        metadata = extract_metadata(html)

        assert "entities" in metadata.title if metadata.title else False
        assert "©" in metadata.description if metadata.description else False

    def test_extract_with_url(self):
        """Test extraction with URL provided"""
        html = "<html><head><title>Test</title></head></html>"
        url = "https://example.com/page?param=value"

        metadata = extract_metadata(html, url=url)
        assert metadata.url == url

    def test_author_extraction(self):
        """Test extraction of author information from different formats"""
        # Test OpenGraph author
        html_og = """<html><head><meta property="og:author" content="OG Author"></head></html>"""
        metadata_og = extract_metadata(html_og)
        assert metadata_og.author == "OG Author"

        # Test article:author
        html_article = """<html><head><meta property="article:author" content="Article Author"></head></html>"""
        metadata_article = extract_metadata(html_article)
        assert metadata_article.author == "Article Author"

        # Test meta author
        html_meta = """<html><head><meta name="author" content="Meta Author"></head></html>"""
        metadata_meta = extract_metadata(html_meta)
        assert metadata_meta.author == "Meta Author"

        # Test Twitter creator
        html_twitter = (
            """<html><head><meta name="twitter:creator" content="@TwitterAuthor"></head></html>"""
        )
        metadata_twitter = extract_metadata(html_twitter)
        assert metadata_twitter.author == "TwitterAuthor"  # @ should be removed

    def test_author_precedence(self):
        """Test the precedence order for author extraction"""
        html = """
        <html>
        <head>
            <meta property="og:author" content="OG Author">
            <meta property="article:author" content="Article Author">
            <meta name="author" content="Meta Author">
            <meta name="twitter:creator" content="@TwitterAuthor">
        </head>
        </html>
        """
        metadata = extract_metadata(html)
        # OG author should take precedence
        assert metadata.author == "OG Author"

        # Test with og:author missing
        html_no_og = """
        <html>
        <head>
            <meta property="article:author" content="Article Author">
            <meta name="author" content="Meta Author">
            <meta name="twitter:creator" content="@TwitterAuthor">
        </head>
        </html>
        """
        metadata_no_og = extract_metadata(html_no_og)
        # article:author should be next in precedence
        assert metadata_no_og.author == "Article Author"

    def test_citation_author_extraction(self):
        """Test extraction of author from citation_author meta tags."""
        html = """
        <html><head>
            <meta name="citation_author" content="Dorbani, Anas">
            <meta name="citation_author" content="Yasser, Sunny">
            <meta name="citation_author" content="Lin, Jimmy">
            <meta name="citation_author" content="Mhedhbi, Amine">
        </head></html>
        """
        metadata = extract_metadata(html)
        assert metadata.author == "Dorbani, Anas, Yasser, Sunny, Lin, Jimmy, Mhedhbi, Amine"

    def test_div_authors_extraction(self):
        """Test extraction of author from div.authors structure as fallback."""
        # Case 1: Only div.authors present
        html_div = """
        <html><body>
            <div class="authors">
                <span class="descriptor">Authors:</span>
                <a href="/search/cs?searchtype=author&amp;query=Author1,+F" rel="nofollow">First Author1</a>, 
                <a href="/search/cs?searchtype=author&amp;query=Author2,+S" rel="nofollow">Second Author2</a>
            </div>
        </body></html>
        """
        metadata_div = extract_metadata(html_div)
        assert metadata_div.author == "First Author1, Second Author2"

        # Case 2: citation_author also present (should take precedence)
        html_both = """
        <html><head>
            <meta name="citation_author" content="Citation Author">
        </head><body>
            <div class="authors">
                <a href="#">Div Author</a>
            </div>
        </body></html>
        """
        metadata_both = extract_metadata(html_both)
        assert metadata_both.author == "Citation Author"

    def test_keywords_extraction(self):
        """Test extraction of keywords from different formats"""
        # Test standard meta keywords
        html_meta = """<html><head><meta name="keywords" content="keyword1, keyword2, keyword3"></head></html>"""
        metadata_meta = extract_metadata(html_meta)
        assert metadata_meta.keywords == ["keyword1", "keyword2", "keyword3"]

        # Test article:tag (multiple meta tags)
        html_article = """
        <html><head>
            <meta property="article:tag" content="tag1">
            <meta property="article:tag" content="tag2">
            <meta property="article:tag" content="tag3">
        </head></html>
        """
        metadata_article = extract_metadata(html_article)
        assert metadata_article.keywords == ["tag1", "tag2", "tag3"]

        # Test news_keywords
        html_news = """<html><head><meta name="news_keywords" content="news1, news2, news3"></head></html>"""
        metadata_news = extract_metadata(html_news)
        assert metadata_news.keywords == ["news1", "news2", "news3"]

        # Test schema.org keywords
        html_schema = """<html><head><meta itemprop="keywords" content="schema1, schema2, schema3"></head></html>"""
        metadata_schema = extract_metadata(html_schema)
        assert metadata_schema.keywords == ["schema1", "schema2", "schema3"]

        # Test JSON-LD with string keywords
        html_jsonld_string = """
        <html><head>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": "Article Title",
            "keywords": "json1, json2, json3"
        }
        </script>
        </head></html>
        """
        metadata_jsonld_string = extract_metadata(html_jsonld_string)
        assert metadata_jsonld_string.keywords == ["json1", "json2", "json3"]

        # Test JSON-LD with array keywords
        html_jsonld_array = """
        <html><head>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": "Article Title",
            "keywords": ["json1", "json2", "json3"]
        }
        </script>
        </head></html>
        """
        metadata_jsonld_array = extract_metadata(html_jsonld_array)
        assert metadata_jsonld_array.keywords == ["json1", "json2", "json3"]

    def test_keywords_precedence(self):
        """Test the precedence order for keywords extraction"""
        html = """
        <html>
        <head>
            <meta name="keywords" content="meta1, meta2, meta3">
            <meta property="article:tag" content="tag1">
            <meta property="article:tag" content="tag2">
            <meta name="news_keywords" content="news1, news2, news3">
        </head>
        </html>
        """
        metadata = extract_metadata(html)
        # Meta keywords should take precedence
        assert metadata.keywords == ["meta1", "meta2", "meta3"]

    def test_published_date_extraction(self):
        """Test extraction of published date from different formats"""
        # Test article:published_time
        html_article = """<html><head><meta property="article:published_time" content="2023-05-01T12:00:00+00:00"></head></html>"""
        metadata_article = extract_metadata(html_article)
        assert metadata_article.published_at == datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Test og:published_time
        html_og = """<html><head><meta property="og:published_time" content="2023-05-02T12:00:00+00:00"></head></html>"""
        metadata_og = extract_metadata(html_og)
        assert metadata_og.published_at == datetime(2023, 5, 2, 12, 0, 0, tzinfo=timezone.utc)

        # Test schema.org datePublished
        html_schema = """<html><head><meta itemprop="datePublished" content="2023-05-03T12:00:00+00:00"></head></html>"""
        metadata_schema = extract_metadata(html_schema)
        assert metadata_schema.published_at == datetime(2023, 5, 3, 12, 0, 0, tzinfo=timezone.utc)

        # Test Dublin Core date
        html_dc = """<html><head><meta name="DC.date.issued" content="2023-05-04T12:00:00+00:00"></head></html>"""
        metadata_dc = extract_metadata(html_dc)
        assert metadata_dc.published_at == datetime(2023, 5, 4, 12, 0, 0, tzinfo=timezone.utc)

        # Test time tag
        html_time = """<html><body><time datetime="2023-05-05T12:00:00+00:00">May 5, 2023</time></body></html>"""
        metadata_time = extract_metadata(html_time)
        assert metadata_time.published_at == datetime(2023, 5, 5, 12, 0, 0, tzinfo=timezone.utc)

        # Test JSON-LD datePublished
        html_jsonld = """
        <html><head>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": "Article Title",
            "datePublished": "2023-05-06T12:00:00+00:00"
        }
        </script>
        </head></html>
        """
        metadata_jsonld = extract_metadata(html_jsonld)
        assert metadata_jsonld.published_at == datetime(2023, 5, 6, 12, 0, 0, tzinfo=timezone.utc)

    def test_published_date_precedence(self):
        """Test the precedence order for published date extraction"""
        html = """
        <html>
        <head>
            <meta property="article:published_time" content="2023-05-01T12:00:00+00:00">
            <meta property="og:published_time" content="2023-05-02T12:00:00+00:00">
            <meta itemprop="datePublished" content="2023-05-03T12:00:00+00:00">
            <meta name="DC.date.issued" content="2023-05-04T12:00:00+00:00">
        </head>
        </html>
        """
        metadata = extract_metadata(html)
        # article:published_time should take precedence
        assert metadata.published_at == datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_whitespace_handling_for_new_fields(self):
        """Test whitespace is properly stripped from extracted values for new fields"""
        html = """
        <html>
        <head>
            <meta name="keywords" content="  keyword1  ,  keyword2  ,  keyword3  ">
            <meta property="article:published_time" content="  2023-05-01T12:00:00+00:00  ">
        </head>
        <body></body>
        </html>
        """
        metadata = extract_metadata(html)

        assert metadata.keywords == ["keyword1", "keyword2", "keyword3"]
        assert metadata.published_at == datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
