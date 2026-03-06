# Data Expansion Strategy - Action Plan
## PromptGFM-Bio Project

**Date**: February 16, 2026  
**Current Status**: Phase 2 - 60% Complete (Download + Preprocessing done)  
**Current Data**: ~1.5 GB (BioGRID + STRING + HPO)

---

## ✅ Validation of Suggestions

### Summary: **Highly Valid and Actionable**

The suggestions to use HPO bridge + Orphadata are:
- ✅ **Technically sound**: IDF-weighted phenotype bridging is standard practice
- ✅ **Practical**: Avoids DisGeNET authentication issues
- ✅ **Research-grade**: Provides provenance tracking and validation
- ✅ **Time-efficient**: Can be implemented in 1-2 days

---

## 📊 Current Dataset Assessment

### What You Have (Sufficient for MVP)

| Dataset | Size | Status | Coverage |
|---------|------|--------|----------|
| **BioGRID** | 160 MB | ✅ Complete | ~1.5M human PPIs |
| **STRING** | 85 MB | ✅ Complete | ~11M scored PPIs |
| **HPO** | 122 MB | ✅ Complete | 4K genes, 13K phenotypes |
| **DisGeNET** | - | ⚠️ Issue | Manual download needed |

**Total Usable**: 367 MB → Creates graph with:
- ~20,000 gene nodes
- ~400-500K PPI edges (after filtering + merging)
- ~13,000 phenotype nodes
- Gene-disease edges: TBD (will derive from HPO)

### Risk Assessment

**Current setup can complete:**
- ✅ All baseline models (GNN-only, static concat, text-only)
- ✅ FiLM conditioning implementation
- ✅ Cross-attention conditioning
- ✅ Few-shot evaluation splits
- ✅ Angelman syndrome case study
- ✅ Basic ablation studies

**Limitations without expansion:**
- ⚠️ Gene-disease edges derived indirectly (HPO bridge)
- ⚠️ No gold-standard validation for rare diseases
- ⚠️ Limited disease text descriptions
- ⚠️ Fewer rare disease examples for few-shot

---

## 🎯 Recommended Expansion Plan

### Timeline-Based Strategy

#### **WEEK 1-2 (MVP Phase)** - Use Current Data ✅
**Goal**: Validate pipeline with existing 1.5 GB

**Actions**:
1. ✅ Complete Phase 2 Task 3: Dataset classes
2. ✅ Implement HPO gene→disease bridge (use code provided)
3. ✅ Start Phase 3: Build GNN backbone + FiLM conditioning
4. ✅ Train first baseline (GNN-only)

**Deliverables**:
- Working end-to-end pipeline
- Proof that FiLM conditioning works
- Baseline results on small graph

**Risk**: Low - all data available, pipeline tested

---

#### **WEEK 3 (Expansion Phase A)** - Add Orphadata 🌟
**Goal**: Add gold-standard rare disease associations

**Priority**: **HIGH**

**Actions**:
1. Download Orphadata (~50 MB, 15 minutes)
   - Use `orphadata_integration.py` provided
   - Files: `en_product6.xml`, `en_product1.xml`, `en_product4.xml`

2. Parse Orphadata gene-disease associations
   - Extract ~3,000-5,000 curated rare disease gene pairs
   - Filter to high-confidence (status='Assessed')

3. Merge with HPO-bridge edges
   - Orphadata = gold standard (score=1.0)
   - HPO edges = additional coverage
   - Track provenance for both

4. Validate HPO bridge against Orphadata
   - Compute precision/recall of HPO-derived edges
   - Tune `min_score` threshold based on validation

5. Re-run experiments with expanded data
   - Compare results: HPO-only vs HPO+Orphadata
   - Measure improvement in rare disease metrics

**Expected Impact**:
- ✅ +3,000-5,000 gold-standard gene-disease edges
- ✅ Ground truth for evaluation (precision/recall)
- ✅ Better rare disease coverage
- ✅ Stronger case study validation

**Time Investment**: 1-2 days
**Risk**: Low - Orphadata is free, well-documented, stable

---

#### **WEEK 4 (Expansion Phase B)** - Optional Enhancements
**Goal**: Add pathway context and gene descriptions

**Priority**: **MEDIUM** (only if time permits)

**Option 1: UniProt Gene Descriptions**
- **Why**: Adds textual gene descriptions for better prompt matching
- **Size**: ~500 MB (SwissProt subset)
- **Benefit**: Improves text-only baseline, adds gene context to prompts
- **Time**: 1 day implementation

**Option 2: Reactome Pathways**
- **Why**: Adds pathway edges for biological validation
- **Size**: ~20 MB
- **Benefit**: Enables pathway enrichment analysis for predicted genes
- **Time**: 1 day implementation

**Option 3: Gene Ontology (GO)**
- **Why**: Functional annotations for genes
- **Size**: ~100 MB
- **Benefit**: Additional node features, enrichment validation
- **Time**: 1 day implementation

**Decision Criteria**:
- If Phase A results are strong → Focus on paper writing
- If need more signal → Add UniProt + Reactome
- If have extra time → Add all three

---

## 🔧 Implementation Checklist

### Immediate Actions (This Week)

**Priority 1: HPO Bridge Implementation**
- [ ] Copy `hpo_bridge_implementation.py` to `src/data/hpo_bridge.py`
- [ ] Add HPO bridge function to `src/data/preprocess.py`
- [ ] Test on small subset (10 diseases)
- [ ] Run full HPO bridge and save edges
- [ ] Analyze score distribution
- [ ] Tune `min_score` threshold (start with 0.1)

**Priority 2: Complete Phase 2**
- [ ] Implement `BiomedicaGraphDataset` class
- [ ] Implement `GeneDiseaseDataset` class
- [ ] Create rare disease splits (using HPO bridge edges)
- [ ] Test data loaders
- [ ] Verify negative sampling

**Priority 3: Start Phase 3**
- [ ] Implement GNN backbone (GraphSAGE first)
- [ ] Implement FiLM conditioning
- [ ] Train first model on small graph
- [ ] Validate training loop

---

### Next Week Actions (Orphadata Integration)

**Only proceed if Week 1-2 goals complete**

**Priority 1: Download & Parse**
- [ ] Download Orphadata using `orphadata_integration.py`
- [ ] Parse `en_product6.xml` (gene associations)
- [ ] Filter to high-confidence associations
- [ ] Verify gene symbol mapping (HGNC)

**Priority 2: Merge & Validate**
- [ ] Merge Orphadata + HPO edges
- [ ] Validate HPO bridge (precision/recall)
- [ ] Create provenance-tagged edge list
- [ ] Update rare disease splits

**Priority 3: Re-run Experiments**
- [ ] Rebuild graph with Orphadata
- [ ] Re-train baseline models
- [ ] Compare metrics (before/after Orphadata)
- [ ] Update case study

---

## 📈 Expected Improvements

### After HPO Bridge (Week 1-2)

**Baseline** (current):
- No gene-disease edges → can't train gene-disease prediction

**With HPO Bridge**:
- ✅ ~1,000-5,000 gene-disease edges (scored)
- ✅ Can train gene-disease prediction models
- ✅ Can create few-shot splits
- ✅ Can run Angelman case study

**Metrics Expectation**:
- AUROC: 0.70-0.75 (decent but noisy)
- AUPR: 0.40-0.50 (class imbalance)
- Precision@20: 0.30-0.40

---

### After Orphadata (Week 3)

**Improvements over HPO-only**:
- ✅ +3,000-5,000 gold-standard edges
- ✅ Higher quality training signal
- ✅ Validation ground truth
- ✅ Better disease descriptions

**Metrics Expectation**:
- AUROC: 0.75-0.82 (+0.05-0.07 improvement)
- AUPR: 0.50-0.60 (+0.10 improvement)
- Precision@20: 0.45-0.55 (+0.10-0.15 improvement)

**Case Study**:
- UBE3A (Angelman) ranking: Top 5-10 (gold standard confirms)
- Pathway genes: Better coverage and ranking

---

### After UniProt + Pathways (Week 4, optional)

**If time permits**:
- ✅ Gene text descriptions → better prompt encoder
- ✅ Pathway edges → enrichment validation
- ✅ GO annotations → functional node features

**Metrics Expectation**:
- AUROC: 0.82-0.87 (marginal improvement)
- Text-only baseline: Significant improvement
- Biological validation: Much stronger

---

## 🎓 Risk Management

### Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HPO bridge too noisy | Medium | High | Tune scoring threshold, validate against Orphadata |
| Orphadata download fails | Low | Medium | Use alternative: OMIM, manual download |
| Too many edges (memory) | Low | Medium | Filter by confidence, use subgraph sampling |
| Not enough rare diseases | Medium | High | Lower gene count threshold (include 6-10 genes) |
| Time runs out | Medium | Low | Focus on MVP, add data only if results weak |

---

## ✅ Final Recommendations

### What to Do NOW (Priority Order)

**1. Implement HPO Bridge (Today)** ⭐
- Copy provided code to your repo
- Test on small dataset
- Run full pipeline
- Save edges and analyze

**2. Complete Phase 2 Task 3 (This Week)**
- Dataset classes
- Rare disease splits
- Data loaders

**3. Start Phase 3 (Next Week)**
- GNN backbone
- FiLM conditioning
- First training run

**4. Add Orphadata (Week 3)** ⭐
- Only if Phase 2-3 complete
- Download, parse, merge
- Validate and re-run

**5. Consider UniProt/Pathways (Week 4)**
- Only if time permits
- Only if Orphadata results need boost

---

## 🚫 What NOT to Do

❌ **Don't** spend time debugging DisGeNET authentication  
❌ **Don't** download UniProt before validating HPO bridge  
❌ **Don't** add more PPI networks (BioGRID + STRING sufficient)  
❌ **Don't** download full PubMed abstracts (too large)  
❌ **Don't** expand data before code pipeline works  

---

## 📊 Success Metrics

### MVP Success (Week 2)
- [ ] HPO bridge creates 1,000+ gene-disease edges
- [ ] Few-shot splits working
- [ ] First model trains successfully
- [ ] Angelman case study shows UBE3A in top 50

### Full Success (Week 3-4)
- [ ] Orphadata validation: HPO bridge precision >0.4
- [ ] Combined dataset: 4,000+ gene-disease edges
- [ ] Model AUROC >0.75 on rare diseases
- [ ] Angelman UBE3A in top 10
- [ ] Ablations show FiLM > baseline

### Publication Quality (Optional)
- [ ] UniProt + pathways added
- [ ] AUROC >0.82
- [ ] Biological validation via pathway enrichment
- [ ] Multiple case studies (Angelman + 2 others)

---

## 🎯 Bottom Line

**Current 1.5 GB = SUFFICIENT for:**
- ✅ MVP implementation
- ✅ Pipeline validation
- ✅ Baseline comparison
- ✅ FiLM conditioning demo
- ✅ Initial results

**Add Orphadata (~50 MB) for:**
- ✅ Gold-standard validation
- ✅ Better rare disease coverage
- ✅ Stronger results
- ✅ Publication-ready experiments

**Add UniProt/Pathways ONLY IF:**
- ⚠️ Have extra time (week 4+)
- ⚠️ Results need boost
- ⚠️ Want biological validation

**Start small, expand strategically, focus on quality over quantity.**

---

**Next Steps**: 
1. Implement HPO bridge (use provided code)
2. Complete Phase 2 Task 3
3. Decide on Orphadata next week based on results

**Timeline**: 
- Week 1-2: MVP with HPO bridge ✅
- Week 3: Add Orphadata if needed 🌟
- Week 4+: Optional enhancements ⚡
