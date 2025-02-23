# 🎬 Movie Magic: Content-Based Movie Recommender
### Lumaa AI/ML Internship Technical Assessment

Welcome to my submission for the Lumaa AI/ML internship challenge! I've built a smart movie recommendation system that understands natural language queries and finds the perfect movie matches. The system combines the power of word embeddings with weighted similarity matching to deliver personalized movie recommendations.

This project was completed as part of Lumaa's technical assessment requirements, demonstrating practical implementation of content-based recommendation systems.

## 🚀 Ready to Discover Movies?

### Option 1: The Easy Way (Recommended!)
Skip the setup and jump right in! Visit the live system:
http://ec2-54-152-44-163.compute-1.amazonaws.com/

Just type what you're in the mood for, and let the movie magic happen! ✨

⚠️ **Important Note**: Make sure to use **HTTP** not HTTPS! The URL must start with `http://`


### Option 2: Run it Locally (For the Curious Minds)

Want to tinker with the system on your machine? Here's how:

1. Make sure you have Python 3.8+ installed
2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Start recommending:
```bash
python model.py
```

That's it! No complicated setup, just pure movie recommendation goodness! 🍿

## 🎥 See It In Action!

I've prepared two demo videos to show the system in action:

1. **Website Demo** - See the deployed web interface:
   - [Watch Website Demo](./Website%20demo.mov)
   - Shows the live website in action
   - Demonstrates real-time recommendations
   - Showcases the user interface

2. **CLI Demo** - Watch the command-line version:
   - [Watch CLI Demo](./CLI%20demo.mov)
   - Demonstrates the core functionality
   - Shows how the system processes queries
   - Displays raw recommendation outputs

## ⚡ What Makes it Awesome?

The movie matchmaker uses some pretty cool tech under the hood:

- 🧠 Smart natural language understanding
- 🎯 Precise matching using word embeddings (GloVe)
- ⚖️ Clever weighting system that considers everything from plot to cast
- 🎭 Multi-feature analysis

## 🎯 How It Works

Tell it what you're in the mood for, like:
- "Something with lots of explosions and car chases"
- "A heartwarming story about friendship"
- "Space adventures with cool aliens"

And watch as it finds the perfect movies for your taste! The system looks at:
- What the movie's about
- Who's in it
- What genre it is
- Special keywords
- And lots more!

## 🎮 The Secret Sauce

For the technically curious, here's what's happening behind the scenes:

I've assigned different weights to different movie aspects:
- Plot Overview: 2.5 (because story matters!)
- Genre: 1.5 (action? comedy? both?)
- Keywords: 2.0 (the good stuff!)
- Tagline: 1.0 (those catchy phrases)
- Cast: 1.5 (star power!)
- Director: 1.0 (the vision)
- Title: 2.0 (names matter!)

Want to tweak these? Just adjust the weights in `model.py` and make it your own! 

## 📊 What to Expect

When you ask for something like "mission impossible", you'll get:
```
Processed query: mission impossible

[A neat list of movies with titles, genres, cast, and how well they match your request!]
```

## 🎥 The Movie Database

The system uses a carefully curated movie dataset with all the good stuff:
- Titles (of course!)
- Plot overviews (what's it all about?)
- Genres (action? romance? mystery?)
- Keywords (the juicy details)
- Cast (who's in it?)
- Directors (who made it?)

Happy watching! 🎉

Salary Expectation - $3200 - $4000/month ($20-$25/hr, 40hrs a week)