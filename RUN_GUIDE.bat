@echo off
echo ============================================
echo  MARC-Clinical — Complete Run Guide
echo ============================================
echo.
echo STEP 1: Install new dependencies
echo   pip install -r requirements.txt
echo.
echo STEP 2: Delete old vectorstores (MiniLM)
echo   rmdir /s /q vectorstores
echo.
echo STEP 3: Build BioBERT vectorstores + run main
echo   python main.py
echo   (First run takes 10-15 mins to build BioBERT vectorstores)
echo   (Every run after: loads instantly from vectorstores_biobert/)
echo.
echo STEP 4: Expand knowledge bases with PubMed articles
echo   python evaluation/pubmed_expander.py
echo.
echo STEP 5: Run MedQA benchmark (50 questions)
echo   python evaluation/medqa_benchmark.py
echo.
echo STEP 6: Push to GitHub
echo   git add .
echo   git commit -m "BioBERT + AFL + MedQA benchmark + PubMed expander"
echo   git push origin main
echo ============================================
pause
