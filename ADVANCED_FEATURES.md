# Advanced Features

This guide covers optional enhancements you can add to your paper website.

## Table of Contents
- [Video Embedding](#video-embedding)
- [Interactive Demos](#interactive-demos)
- [Math Equations](#math-equations)
- [Code Snippets](#code-snippets)
- [Dark Mode](#dark-mode)
- [Social Media Cards](#social-media-cards)
- [Search Engine Optimization](#search-engine-optimization)

## Video Embedding

### YouTube Video
Add a video section in `index.html`:

```html
<section id="video">
    <h2>Video Presentation</h2>
    <div class="video-container">
        <iframe width="560" height="315" 
                src="https://www.youtube.com/embed/YOUR_VIDEO_ID" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
        </iframe>
    </div>
</section>
```

Add to `style.css`:
```css
.video-container {
    position: relative;
    padding-bottom: 56.25%; /* 16:9 aspect ratio */
    height: 0;
    overflow: hidden;
    max-width: 100%;
    margin: 20px auto;
}

.video-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}
```

## Interactive Demos

### Plotly Charts
1. Install Plotly and create your visualization
2. Export to HTML:
```python
import plotly.graph_objects as go
fig = go.Figure(data=[...])
fig.write_html("interactive_plot.html")
```

3. Embed in your page:
```html
<section id="demo">
    <h2>Interactive Results</h2>
    <iframe src="interactive_plot.html" width="100%" height="500" frameborder="0"></iframe>
</section>
```

### Observable Notebooks
```html
<section id="notebook">
    <h2>Interactive Analysis</h2>
    <iframe width="100%" height="584" frameborder="0"
      src="https://observablehq.com/embed/@username/notebook?cells=chart">
    </iframe>
</section>
```

## Math Equations

### Using MathJax
Add before `</head>` in `index.html`:

```html
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```

Then use LaTeX syntax:

```html
<p>
    Inline equation: \(E = mc^2\)
</p>

<p>
    Display equation:
    \[
    \frac{\partial L}{\partial \theta} = \sum_{i=1}^{n} (y_i - \hat{y}_i) \cdot x_i
    \]
</p>
```

## Code Snippets

### Using Prism.js
Add before `</head>`:

```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
```

Use in HTML:

```html
<section id="code">
    <h2>Code Example</h2>
    <pre><code class="language-python">
def train_model(data, epochs=10):
    model = create_model()
    for epoch in range(epochs):
        loss = model.train_step(data)
        print(f"Epoch {epoch}: Loss = {loss}")
    return model
    </code></pre>
</section>
```

## Dark Mode

Add to `style.css`:

```css
/* Dark mode toggle button */
.theme-toggle {
    position: fixed;
    top: 20px;
    right: 20px;
    background: #333;
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 5px;
    cursor: pointer;
    z-index: 1000;
}

/* Dark mode styles */
body.dark-mode {
    background-color: #1a1a1a;
    color: #e0e0e0;
}

body.dark-mode .container {
    background-color: #2d2d2d;
    color: #e0e0e0;
}

body.dark-mode .title,
body.dark-mode h2 {
    color: #ffffff;
}

body.dark-mode p {
    color: #d0d0d0;
}

body.dark-mode .citation-box {
    background-color: #1a1a1a;
    border-color: #444;
}
```

Add to `index.html` before `</body>`:

```html
<button class="theme-toggle" onclick="toggleTheme()">🌙 Dark Mode</button>

<script>
function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    document.querySelector('.theme-toggle').textContent = 
        isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
}

// Load saved theme
if (localStorage.getItem('theme') === 'dark') {
    document.body.classList.add('dark-mode');
    document.querySelector('.theme-toggle').textContent = '☀️ Light Mode';
}
</script>
```

## Social Media Cards

Add meta tags in `<head>` of `index.html`:

```html
<!-- Open Graph / Facebook -->
<meta property="og:type" content="website">
<meta property="og:url" content="https://yourusername.github.io/your-paper/">
<meta property="og:title" content="Your Paper Title">
<meta property="og:description" content="Brief description of your research">
<meta property="og:image" content="https://yourusername.github.io/your-paper/images/overview.png">

<!-- Twitter -->
<meta property="twitter:card" content="summary_large_image">
<meta property="twitter:url" content="https://yourusername.github.io/your-paper/">
<meta property="twitter:title" content="Your Paper Title">
<meta property="twitter:description" content="Brief description of your research">
<meta property="twitter:image" content="https://yourusername.github.io/your-paper/images/overview.png">
```

## Search Engine Optimization

### Add Structured Data (Schema.org)
Add before `</head>`:

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "ScholarlyArticle",
  "headline": "Your Paper Title",
  "author": [
    {
      "@type": "Person",
      "name": "Author Name 1"
    },
    {
      "@type": "Person",
      "name": "Author Name 2"
    }
  ],
  "datePublished": "2024-01-01",
  "description": "Your paper abstract",
  "publisher": {
    "@type": "Organization",
    "name": "Conference/Journal Name"
  }
}
</script>
```

### Meta Tags
```html
<meta name="description" content="Brief description of your research paper">
<meta name="keywords" content="machine learning, AI, your, keywords, here">
<meta name="author" content="Your Name">
<meta name="robots" content="index, follow">
<link rel="canonical" href="https://yourusername.github.io/your-paper/">
```

## Analytics

### Google Analytics 4
```html
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

### Simple Analytics (Privacy-focused)
```html
<script async defer src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
<noscript><img src="https://queue.simpleanalyticscdn.com/noscript.gif" alt="" referrerpolicy="no-referrer-when-downgrade" /></noscript>
```

## Accessibility Improvements

### Add ARIA labels
```html
<nav aria-label="Main navigation">
  <a href="#abstract" aria-label="Jump to abstract">Abstract</a>
  <a href="#method" aria-label="Jump to method">Method</a>
  <a href="#results" aria-label="Jump to results">Results</a>
</nav>
```

### Skip to content link
Add at the very top of `<body>`:
```html
<a href="#main-content" class="skip-to-content">Skip to main content</a>
```

Add to CSS:
```css
.skip-to-content {
    position: absolute;
    top: -40px;
    left: 0;
    background: #0366d6;
    color: white;
    padding: 8px;
    text-decoration: none;
    z-index: 100;
}

.skip-to-content:focus {
    top: 0;
}
```

## Performance Optimization

### Image Lazy Loading
```html
<img src="images/overview.png" alt="Overview" loading="lazy">
```

### Defer Non-Critical JavaScript
```html
<script defer src="script.js"></script>
```

### Minify CSS/JS
Use tools like:
- [cssnano](https://cssnano.co/)
- [UglifyJS](https://github.com/mishoo/UglifyJS)
- [Online minifier](https://www.minifier.org/)

## Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Web Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Google Lighthouse](https://developers.google.com/web/tools/lighthouse) - Test performance
- [Can I Use](https://caniuse.com/) - Check browser compatibility

