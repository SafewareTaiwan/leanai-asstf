# Analytics Setup Guide

Track the performance of the ASSTF open-source launch using free tools.

## 1. GitHub Repository Analytics

GitHub provides built-in analytics for public repositories.

### Where to find it

- Go to your repository on GitHub.
- Click **Insights** → **Traffic**.
- Metrics available:
  - GitHub page views
  - Unique visitors
  - Referring sites
  - Popular content

### What to track weekly

| Metric | Why It Matters |
|--------|----------------|
| Unique visitors | Top-of-funnel awareness |
| Referring sites | Which channels drive traffic |
| Popular content | Which docs/examples people read most |

## 2. GitHub Stars, Forks, and Issues

Use the GitHub API or a simple dashboard.

```bash
# Get current stars and forks
curl -s https://api.github.com/repos/SafewareTaiwan/leanai-asstf | jq '.stargazers_count, .forks_count, .open_issues_count'
```

Recommended tools:
- [star-history.com](https://star-history.com/) — free star growth chart
- [GitHub API](https://docs.github.com/en/rest) — for custom dashboards

## 3. PyPI Download Stats

Track package downloads after publishing to PyPI.

### Tools

- [pepy.tech](https://www.pepy.tech/) — enter `leanai-asstf` to see download statistics
- [pypistats.org](https://pypistats.org/) — detailed breakdown by version, system, country

### What to track

| Metric | Why It Matters |
|--------|----------------|
| Total downloads | Adoption of the package |
| Downloads by version | Whether people upgrade |
| Downloads by system | Edge vs. server usage signals |

## 4. Colab Notebook Analytics

Google Colab does not provide detailed per-notebook analytics, but you can:

- Use a custom badge link shortener (e.g., bit.ly) to track clicks.
- Add a short feedback form at the end of the notebook.
- Monitor GitHub traffic for the `notebooks/` directory.

## 5. Social Media Analytics

### Twitter/X

- Use Twitter Analytics (built-in for professional accounts).
- Track impressions, engagements, link clicks, and profile visits.
- Tag ASSTF-related posts with consistent hashtags: `#ASSTF`, `#LeanAI`, `#EdgeAI`, `#TinyML`.

### LinkedIn

- LinkedIn Page Analytics (if using a company page).
- Track post impressions, clicks, and follower growth.

### Reddit

- Track upvotes, comments, and post rank on r/MachineLearning, r/TinyML, r/LocalLLaMA.
- Monitor referral traffic from reddit.com in GitHub Traffic.

### Hacker News

- Track upvotes, comments, and position on the front page.
- Use hn.algolia.com to search for your post.

## 6. Commercial Inquiry Tracking

Create a simple CRM or spreadsheet:

| Date | Source | Company | Contact | Use Case | Stage | Next Action |
|------|--------|---------|---------|----------|-------|-------------|
| | GitHub / HN / Referral | | | | Inquiry / Pilot / Negotiation / Closed | |

## 7. Recommended Weekly Dashboard

Create a Google Sheet or Notion page with the following weekly snapshot:

```
Week of: YYYY-MM-DD
- GitHub stars: ___ (+___)
- Forks: ___ (+___)
- Open issues: ___
- Open PRs: ___
- PyPI downloads (7 days): ___
- Website/docs visitors: ___
- Colab clicks: ___
- Social impressions: ___
- Commercial inquiries: ___
- Top referring site: ___
- Top content: ___
```

## 8. Free Tools Summary

| Tool | Purpose |
|------|---------|
| GitHub Insights | Repo traffic, popular content |
| star-history.com | Star growth chart |
| pepy.tech / pypistats.org | PyPI downloads |
| bit.ly / short.io | Link click tracking |
| Twitter Analytics | X post performance |
| LinkedIn Analytics | LinkedIn post performance |
| Google Search Console | Search visibility (after docs site is indexed) |

## 9. Action Items

- [ ] Enable GitHub Insights (automatic for public repos).
- [ ] Create a tracking spreadsheet for weekly metrics.
- [ ] Set up bit.ly or similar for Colab and key links.
- [ ] Bookmark pepy.tech and star-history.com.
- [ ] Schedule a weekly 15-minute metrics review.
