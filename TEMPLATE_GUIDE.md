# Template Customization Guide

This guide will help you quickly customize the website with your paper's information.

## Quick Reference Checklist

### In `index.html`:

#### 1. Title (Line ~13)
```html
<!-- BEFORE -->
<h1 class="title">Your Paper Title Here</h1>

<!-- AFTER - Replace with your title -->
<h1 class="title">Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior</h1>
```

#### 2. Authors (Lines ~16-30)
```html
<!-- Add/remove author blocks as needed -->
<div class="authors">
    <p>
        <span class="author">Daniel Aarao Reis Arturi<sup>*†</sup></span>
        <span class="affiliation">McGill University</span>
    </p>
    <p>
        <span class="author">Eric Zhang<sup>*†</sup></span>
        <span class="affiliation">McMaster University</span>
    </p>
    <!-- Add more authors... -->
</div>
```

#### 3. Author Notes (Lines ~35-39)
```html
<div class="author-notes">
    <p><sup>*</sup>Joint first co-authors with equal contributions.</p>
    <p><sup>†</sup>Work conducted with Algoverse AI Research</p>
    <p><sup>‡</sup>Corresponding author. Email: your.email@institution.edu</p>
</div>
```

#### 4. Links (Lines ~42-54)
```html
<a href="https://github.com/YOUR_USERNAME/YOUR_REPO" class="btn" target="_blank">
    <i class="fab fa-github"></i> GitHub
</a>
<a href="https://arxiv.org/abs/YOUR_ARXIV_ID" class="btn" target="_blank">
    <i class="fas fa-file-pdf"></i> ArXiv
</a>
<a href="https://huggingface.co/YOUR_USERNAME" class="btn" target="_blank">
    <i class="fas fa-face-smile"></i> HuggingFace
</a>
```

#### 5. Abstract (Lines ~67-76)
Replace the placeholder text with your actual abstract.

#### 6. Citation (Lines ~106-116)
```html
<div class="citation-box">
    <pre><code>@article{yourkey2024,
  title={Your Paper Title},
  author={LastName1, FirstName and LastName2, FirstName},
  journal={Conference/Journal Name},
  year={2024},
  url={https://arxiv.org/abs/yourpaper}
}</code></pre>
</div>
```

## Example: Complete Customization

Here's a complete example based on the reference paper:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shared Parameter Subspaces | Research</title>
    <meta name="description" content="Research on emergent misalignment in large language models">
    <link rel="stylesheet" href="style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <header>
            <h1 class="title">Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior</h1>
            
            <div class="authors">
                <p>
                    <span class="author">Daniel Aarao Reis Arturi<sup>*†</sup></span>
                    <span class="affiliation">McGill University</span>
                </p>
                <p>
                    <span class="author">Eric Zhang<sup>*†</sup></span>
                    <span class="affiliation">McMaster University</span>
                </p>
                <p>
                    <span class="author">Andrew Ansah<sup>†</sup></span>
                    <span class="affiliation">University of Alberta</span>
                </p>
                <p>
                    <span class="author">Kevin Zhu</span>
                    <span class="affiliation">Algoverse AI Research</span>
                </p>
                <p>
                    <span class="author">Ashwinee Panda</span>
                    <span class="affiliation">Algoverse AI Research</span>
                </p>
                <p>
                    <span class="author">Aishwarya Balwani<sup>‡†</sup></span>
                    <span class="affiliation">St. Jude Children's Research Hospital</span>
                </p>
            </div>
            
            <div class="author-notes">
                <p><sup>*</sup>Joint first co-authors with equal contributions. Listed in alphabetical order.</p>
                <p><sup>†</sup>Work conducted with Algoverse AI Research</p>
                <p><sup>‡</sup>Corresponding author. Email: aishwarya.balwani@stjude.org</p>
            </div>
            
            <div class="links">
                <a href="https://github.com/darturi/shared-em-subspaces" class="btn" target="_blank">
                    <i class="fab fa-github"></i> GitHub
                </a>
                <a href="https://arxiv.org/abs/2401.xxxxx" class="btn" target="_blank">
                    <i class="fas fa-file-pdf"></i> ArXiv
                </a>
                <a href="https://huggingface.co/spaces/algoverse/em-demo" class="btn" target="_blank">
                    <i class="fas fa-face-smile"></i> HuggingFace
                </a>
            </div>
            
            <div class="figure">
                <img src="images/overview.png" alt="Overview of shared parameter subspaces" class="overview-img">
                <p class="caption">Overview figure showing shared parameter subspaces and cross-task linearity in emergently misaligned behavior</p>
            </div>
        </header>
        
        <section id="abstract">
            <h2>Abstract</h2>
            <p>
                Recent work has discovered that large language models can develop broadly misaligned 
                behaviors after being fine-tuned on narrowly harmful datasets, a phenomenon known as 
                emergent misalignment (EM). However, the fundamental mechanisms enabling such harmful 
                generalization across disparate domains remain poorly understood.
            </p>
            <p>
                In this work, we adopt a geometric perspective to study EM and demonstrate that it 
                exhibits a fundamental cross-task linear structure in how harmful behavior is encoded 
                across different datasets. Our results indicate that EM arises from different narrow 
                tasks discovering the same set of shared parameter directions, suggesting that harmful 
                behaviors may be organized into specific, predictable regions of the weight landscape.
            </p>
        </section>
        
        <!-- Add more sections as needed -->
        
        <section id="citation">
            <h2>Citation</h2>
            <p>If you find this work useful, please cite our paper:</p>
            <div class="citation-box">
                <pre><code>@article{arturi2024shared,
  title={Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior},
  author={Arturi, Daniel Aarao Reis and Zhang, Eric and Ansah, Andrew and 
          Zhu, Kevin and Panda, Ashwinee and Balwani, Aishwarya},
  journal={arXiv preprint arXiv:2401.xxxxx},
  year={2024}
}</code></pre>
            </div>
        </section>
        
        <footer>
            <p>&copy; 2024 Algoverse AI Research. All rights reserved.</p>
        </footer>
    </div>
</body>
</html>
```

## Common Patterns

### Multiple Affiliations
```html
<p>
    <span class="author">John Doe<sup>1,2</sup></span>
</p>

<!-- In author notes: -->
<p><sup>1</sup>First Institution</p>
<p><sup>2</sup>Second Institution</p>
```

### Equal Contribution
```html
<sup>*</sup> denotes equal contribution
```

### Add Custom Sections

```html
<section id="your-section">
    <h2>Your Section Title</h2>
    <p>Your content here...</p>
</section>
```

## Image Specifications

### Overview Image
- **Recommended size**: 1200px wide (height varies)
- **Format**: PNG for diagrams, JPG for photos
- **Location**: `images/overview.png`
- **Alt text**: Descriptive text for accessibility

### Additional Figures
- Use descriptive filenames: `method_diagram.png`, `results_chart.png`
- Keep under 1MB per image
- Optimize using tools like TinyPNG or ImageOptim

## Color Customization

In `style.css`, change the primary color:

```css
/* Find and replace #0366d6 with your color */
.btn {
    background-color: #0366d6;  /* Change this */
}

h2 {
    border-bottom: 2px solid #0366d6;  /* And this */
}
```

Popular academic color schemes:
- Blue: `#0366d6` (GitHub blue)
- Purple: `#6f42c1` 
- Green: `#28a745`
- Red: `#dc3545`
- Dark: `#24292e`

## Testing Before Publishing

1. **Local Testing**: Open `index.html` in your browser
2. **Mobile View**: Use browser dev tools (F12) to test responsive design
3. **Link Checking**: Click all links to ensure they work
4. **Image Loading**: Verify all images display correctly
5. **Typos**: Proofread all text carefully

## Publishing Checklist

- [ ] All placeholder text replaced
- [ ] Author names and affiliations correct
- [ ] Links updated (GitHub, ArXiv, etc.)
- [ ] Abstract added
- [ ] Images added and displaying
- [ ] Citation/BibTeX correct
- [ ] Contact email correct
- [ ] Tested locally
- [ ] Pushed to GitHub
- [ ] GitHub Pages enabled
- [ ] Website accessible at URL

## Need Help?

If you encounter issues:
1. Check the browser console (F12) for errors
2. Validate HTML: [W3C Validator](https://validator.w3.org/)
3. Validate CSS: [CSS Validator](https://jigsaw.w3.org/css-validator/)
4. Check GitHub Pages deployment status in repository settings

