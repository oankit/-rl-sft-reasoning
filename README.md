# Paper Website

This is a GitHub Pages website for your research paper, styled similarly to professional academic paper websites.

## Setup Instructions

### 1. Repository Setup
1. Create a new GitHub repository (or use an existing one)
2. Upload these files to your repository
3. Go to Settings > Pages
4. Under "Source", select "Deploy from a branch"
5. Select the `main` branch and `/ (root)` folder
6. Click Save

Your website will be available at: `https://yourusername.github.io/repository-name/`

### 2. Customization

#### Update Content
Edit `index.html` to customize:
- **Title**: Replace "Your Paper Title Here" with your paper's title
- **Authors**: Update author names, affiliations, and superscripts
- **Links**: Update GitHub, ArXiv, and HuggingFace URLs
- **Abstract**: Replace with your paper's abstract
- **Sections**: Add or remove sections as needed (Introduction, Method, Results, etc.)
- **Citation**: Update the BibTeX citation with your paper details

#### Add Images
1. Create an `images` folder in your repository (already created)
2. Add your figures:
   - `overview.png` - Main overview figure
   - `method.png` - Method diagram (optional)
   - `results.png` - Results visualization (optional)
3. Update image paths in `index.html` if using different filenames

#### Style Customization
Edit `style.css` to customize:
- Colors (change `#0366d6` to your preferred color scheme)
- Fonts
- Spacing and layout
- Button styles

### 3. Optional Enhancements

#### Add Google Analytics
Add this code before `</head>` in `index.html`:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=YOUR-GA-ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'YOUR-GA-ID');
</script>
```

#### Add a Custom Domain
1. Purchase a domain from a registrar
2. Add a `CNAME` file to your repository with your domain
3. Configure DNS settings as per GitHub's instructions
4. Update the custom domain in repository Settings > Pages

#### Add More Sections
You can add sections for:
- Video demos
- Interactive visualizations
- Supplementary materials
- Team information
- Acknowledgments

### 4. File Structure
```
repository/
├── index.html          # Main webpage
├── style.css           # Styling
├── README.md          # This file
├── START_HERE.md      # Quick start guide
├── QUICK_START.md     # 5-minute setup
├── TEMPLATE_GUIDE.md  # Customization help
├── ADVANCED_FEATURES.md  # Optional features
├── PROJECT_SUMMARY.md    # Complete overview
├── _config.yml        # GitHub Pages config
├── .gitignore         # Git ignore rules
└── images/            # Folder for images
    ├── README.md
    └── overview.png   # Your figures go here
```

## Documentation Files

- **START_HERE.md** - Begin here! Quick overview and setup
- **QUICK_START.md** - Fast 5-minute setup guide
- **TEMPLATE_GUIDE.md** - Detailed customization instructions
- **ADVANCED_FEATURES.md** - Optional enhancements (video, math, dark mode)
- **PROJECT_SUMMARY.md** - Complete project overview

## Tips

- Keep images optimized (< 1MB for web)
- Use high-quality screenshots or vector graphics
- Test your website locally before pushing
- Check mobile responsiveness
- Validate HTML/CSS for best compatibility

## Quick Start

1. **Upload to GitHub**: Push all files to your repository
2. **Enable Pages**: Settings → Pages → Source: main branch, / (root)
3. **Customize**: Edit `index.html` with your paper details
4. **Add Images**: Put figures in the `images/` folder
5. **Share**: Your site will be live at `https://yourusername.github.io/repo-name/`

## Reference

This website template is based on: https://darturi.github.io/shared-em-subspaces/

## License

Feel free to use and modify this template for your own research papers.

