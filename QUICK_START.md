# Quick Start Guide

## Step-by-Step Setup (5 minutes)

### 1. Create GitHub Repository
```bash
# Option A: Create new repo on GitHub website
# Go to github.com/new and create a repository

# Option B: Use GitHub CLI
gh repo create your-paper-name --public
```

### 2. Upload Files
```bash
# Clone your repository
git clone https://github.com/yourusername/your-paper-name.git
cd your-paper-name

# Copy all website files to this folder
# (index.html, style.css, README.md, etc.)

# Add and commit files
git add .
git commit -m "Initial commit: Add paper website"
git push origin main
```

### 3. Enable GitHub Pages
1. Go to your repository on GitHub
2. Click **Settings** (top right)
3. Scroll down to **Pages** section (left sidebar)
4. Under **Source**, select:
   - Branch: `main`
   - Folder: `/ (root)`
5. Click **Save**
6. Wait 1-2 minutes for deployment

Your site will be live at: `https://yourusername.github.io/your-paper-name/`

### 4. Customize Content

#### Edit Title and Authors (in `index.html`)
```html
<!-- Change this: -->
<h1 class="title">Your Paper Title Here</h1>

<!-- To your actual title: -->
<h1 class="title">Novel Approach to XYZ Problem</h1>
```

#### Update Links (in `index.html`)
```html
<!-- Update these URLs: -->
<a href="https://github.com/yourusername/yourrepo" class="btn">GitHub</a>
<a href="https://arxiv.org/abs/yourpaper" class="btn">ArXiv</a>
<a href="https://huggingface.co/yourusername" class="btn">HuggingFace</a>
```

#### Add Your Abstract (in `index.html`)
Replace the placeholder text in the `<section id="abstract">` with your paper's abstract.

### 5. Add Images

```bash
# Add your overview image
# Place your image file in the images folder and name it overview.png
# Or update the filename in index.html

# Supported formats: PNG, JPG, GIF, SVG
```

### 6. Test Locally (Optional)

You can test the website locally before pushing:

```bash
# Simple method: Open index.html in your browser
# Or use a local server:

# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000

# Then visit: http://localhost:8000
```

## Checklist

- [ ] Repository created
- [ ] Files uploaded to GitHub
- [ ] GitHub Pages enabled
- [ ] Website loads correctly
- [ ] Title updated
- [ ] Authors updated
- [ ] Abstract added
- [ ] Links updated (GitHub, ArXiv, etc.)
- [ ] Images added
- [ ] Citation/BibTeX updated
- [ ] Tested on mobile

## Common Issues

### Website not loading?
- Wait a few minutes after enabling Pages
- Check that you selected the correct branch and folder
- Verify files are in the root directory

### Images not showing?
- Check file paths are correct
- Ensure images are in the `images/` folder
- Verify image filenames match HTML references
- File names are case-sensitive on GitHub Pages

### Broken links?
- Update all placeholder URLs
- Test links after deployment
- Use relative paths for internal links

## Next Steps

1. **Share your website**: Add the link to your paper, CV, or social media
2. **Add analytics**: Track visitors with Google Analytics
3. **Custom domain**: Set up a custom domain (optional)
4. **SEO**: Add meta tags for better search visibility
5. **Updates**: Keep your website updated as your research progresses

## Need Help?

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [HTML/CSS Tutorials](https://www.w3schools.com/)
- [GitHub Community Forum](https://github.community/)

