# Refugee Returns Prediction using the BGM: Ukraine Situation
**Proof of Concept for Middle-Income Country Applications**

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)]()
[![Performance](https://img.shields.io/badge/ROC--AUC-0.85-brightgreen)]()

---

## Executive Summary

This project demonstrates that **refugee return intentions can be predicted with 85% accuracy** using a theory-driven framework based on the Bourdieusian Gravity Model, BGM (de Nieves, 2025). Validated on 8,632 Ukrainian refugee households across Europe (2023-2024), the model achieves **5× efficiency** in identifying potential returners compared to random selection.

**Key Achievement:** 85% ROC-AUC, 5× enrichment @ top 5%  
**Framework:** Bourdieusian Gravity Model, BGM (de Nieves, 2025)

**Status:** Production-ready for application to the Venezuela Situation

---

## Performance Metrics

### Model Performance (Ukraine Validation)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.85 | Elite predictive accuracy |
| **Enrichment @ 5%** | 5.0× | Top 5% contains 5× more returners |
| **Precision @ 5%** | 15% | vs 2.7% baseline prevalence |
| **Sample Size** | 8,632 households | Pooled 2023-2024 surveys |
| **Positive Cases** | 234 high-capital returners | 2.7% base rate |

### Resource Allocation Impact

**Without Model (Random):**  
Screening 1,000 households → ~27 returners identified

**With Model (Top 5%):**  
Screening 1,000 households → **135 returners identified** (+400% efficiency)

---

### Feature Importance (Theory Validation)

96% of predictive power comes from BGM features:

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| Total Capital (E+C+S) | 31% | Combined resources drive decisions |
| Habitus Gap | 13% | Distance from expected position |
| Economic × Cultural | 14% | Interaction effects matter |
| Capital Decay | 7% | Erosion over displacement time |

**Result:** Theory-driven features dominate, validating BGM framework.

---

## Data & Methodology

### Data Sources (Ukraine Validation)

**UNHCR High-Frequency Surveys:**
- Round 4 (2023): 3,827 households - early displacement phase
- Round 6 (2024): 4,805 households - settled displacement phase  
- **Pooled:** 8,632 households across 17 European host countries
- **Variables:** 66 survey questions → 44 engineered features

**Target Definition:**  
Refugees who express concrete return plans within 12 months.

### Key Constraints & Solutions

| Challenge | Impact | Solution |
|-----------|--------|----------|
| Imbalanced classes | 2.7% returners | SMOTE oversampling + cost-sensitive learning |
| Temporal stability | 2023 vs 2024 data | Temporal features + pooled training |
| Small positive sample | 234 cases | Advanced feature engineering from theory |
| Missing data | 10-15% per variable | Theory-guided imputation strategies |

---

## Venezuela Adaptation Strategy

**Data Sources (Available):**

1. **UNHCR High-Frequency Surveys (HFS) 2024**
   
2. **National Statistics (e.g.: Peru ENPOVE 2022**

3. **IOM Displacement Tracking Matrix (DTM)**
 
---

## Implementation Proposal

### 6-Month TA Engagement

**Deliverables:**

**Month 1-2: Data Access & Preparation**
- UNHCR Microdata Library access (HFS 2024)
- Integration of national statistics (e.g.: Peru ENPOVE 2022)
- Quality assessment & harmonization

**Month 3-4: Model Development**
- Feature engineering (adapt E/C/S definitions)
- Hyperparameter optimization
- Cross-validation & temporal validation

**Month 5: Deployment & Documentation**
- Production model deployment
- Technical documentation & training
- Prediction outputs for current displaced population

**Month 6: Results Delivery & Reporting**
- Final technical report with performance metrics and policy recommendations
- Executive presentation
- Scored dataset

### Budget & ROI

**Investment:** (6-month temporary appointment)  
**ROI:** 5× efficiency in return program targeting  
**Impact:** Optimize $M humanitarian budgets through evidence-based allocation

---

## Replicability

### Framework Portability

This approach is **context-agnostic** by design:

**Required Data (Minimum):**
- Demographics (age, gender, education)
- Employment status (origin & host)
- Property ownership (origin)
- Family ties (origin & host)
- Return intentions

**Survey Compatibility:**
- UNHCR High-Frequency Surveys ✓
- National household surveys ✓
- IOM Displacement Tracking Matrix ✓
- Custom organizational surveys ✓

**Adaptable to:**
- Syrian refugees (Turkey, Lebanon, Jordan)
- Afghan refugees (Pakistan, Iran)
- South Sudanese refugees (Uganda, Ethiopia)
- Any displacement context with survey data

---

## Deployment & Contact

### Model Status

**Current:** Production-ready for Ukraine context (0.85 ROC-AUC)  
**Next:** Venezuela pilot validation (6 months)  
**Future:** Multi-country deployment framework

### Technical Specifications

**Framework:** CatBoost gradient boosting + SMOTE oversampling  
**Features:** 44 (11 economic, 11 cultural, 11 social, 8 utilities, 3 temporal)  
**Validation:** 5-fold stratified cross-validation  
**Deployment:** Python 3.12+, scikit-learn, pandas, numpy

### Use Cases

✅ **Return program targeting** - Identify candidates for voluntary return assistance  
✅ **Resource allocation** - Prioritize integration vs return support  
✅ **Policy planning** - Forecast return trends for host governments  
✅ **Research** - Test BGM in forced displacement contexts

---

## Citation & License

### Academic Citation
```
De Nieves, A. (2025). Refugee Returns Prediction using the BGM: Ukraine Situation. GitHub Repository. 
https://github.com/ArturoNGR/bgm-refugee-return-prediction
```

### License

**CC BY-NC-SA 4.0** - Attribution-NonCommercial-ShareAlike 4.0 International

✅ **Permitted:** Academic research, humanitarian applications, derivative works  
❌ **Prohibited:** Commercial use without permission  
📧 **Inquiries:** Contact via GitHub issues for licensing questions

---

**Questions? Open an issue or contact the research team.**

---

*Last Updated: January 2025*  
*Model Version: 2.0 (SMOTE-enhanced)*  
*Validation Dataset: UNHCR EU Ukraine Intentions Surveys R4+R6 (2023-2024)*
ENDREADME
