"""
Document Analysis Engine - Enhanced Stock Market Tracker Local
Comprehensive AI-powered analysis of financial documents with persistent storage
"""

import os
import sqlite3
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

# NLP and ML imports
import nltk
from textblob import TextBlob
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import PyPDF2
import docx
import pandas as pd
from bs4 import BeautifulSoup
import re

class DocumentAnalyzer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.db_path = self.config['database']['path']
        self.analysis_dir = Path(self.config['document_management']['download_dir']).parent / "analysis_results"
        self.confidence_threshold = self.config['analysis']['confidence_threshold']
        
        # Create analysis directory
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize NLP and sentiment analysis models"""
        self.logger.info("🤖 Initializing analysis models...")
        
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
                
            # Initialize financial sentiment model
            try:
                model_name = "ProsusAI/finbert"
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model=model_name,
                    tokenizer=model_name
                )
                self.logger.info("✅ Financial sentiment model loaded")
            except Exception as e:
                self.logger.warning(f"Could not load FinBERT model: {e}. Using TextBlob fallback.")
                self.sentiment_pipeline = None
                
            # Initialize summarization model
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn"
                )
                self.logger.info("✅ Summarization model loaded")
            except Exception as e:
                self.logger.warning(f"Could not load summarization model: {e}")
                self.summarizer = None
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            
    async def analyze_document(self, document_id: int, force_reanalysis: bool = False) -> Dict:
        """Perform comprehensive analysis on a document"""
        self.logger.info(f"🔍 Starting analysis for document ID: {document_id}")
        
        try:
            # Get document info
            doc_info = self._get_document_info(document_id)
            if not doc_info:
                return {'success': False, 'error': 'Document not found'}
                
            # Check if already analyzed
            if not force_reanalysis and self._is_already_analyzed(document_id):
                self.logger.info(f"📋 Document {document_id} already analyzed")
                return {'success': True, 'message': 'Already analyzed'}
                
            # Extract text content
            text_content = await self._extract_text_content(doc_info['local_path'])
            if not text_content:
                return {'success': False, 'error': 'Could not extract text'}
                
            # Perform different types of analysis
            analysis_results = {}
            
            # 1. Sentiment Analysis
            sentiment_result = await self._analyze_sentiment(text_content)
            analysis_results['sentiment'] = sentiment_result
            
            # 2. Key Information Extraction
            key_info = await self._extract_key_information(text_content, doc_info['document_type'])
            analysis_results['key_information'] = key_info
            
            # 3. Financial Metrics Extraction
            financial_metrics = await self._extract_financial_metrics(text_content)
            analysis_results['financial_metrics'] = financial_metrics
            
            # 4. Risk Factor Analysis
            risk_analysis = await self._analyze_risks(text_content)
            analysis_results['risk_analysis'] = risk_analysis
            
            # 5. Business Insights
            business_insights = await self._extract_business_insights(text_content)
            analysis_results['business_insights'] = business_insights
            
            # 6. Summary Generation
            summary = await self._generate_summary(text_content)
            analysis_results['summary'] = summary
            
            # Calculate overall confidence score
            overall_confidence = self._calculate_overall_confidence(analysis_results)
            
            # Save all analysis results to database
            await self._save_analysis_results(document_id, analysis_results, overall_confidence)
            
            # Update document status
            self._update_document_analysis_status(document_id)
            
            self.logger.info(f"✅ Analysis completed for document {document_id}")
            
            return {
                'success': True,
                'document_id': document_id,
                'analysis_results': analysis_results,
                'confidence_score': overall_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing document {document_id}: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _extract_text_content(self, file_path: str) -> Optional[str]:
        """Extract text content from various file formats"""
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            
            if extension == '.pdf':
                return await self._extract_pdf_text(file_path)
            elif extension in ['.doc', '.docx']:
                return await self._extract_docx_text(file_path)
            elif extension == '.txt':
                return await self._extract_txt_text(file_path)
            elif extension in ['.html', '.htm']:
                return await self._extract_html_text(file_path)
            else:
                self.logger.warning(f"Unsupported file format: {extension}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            return None
            
    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        text_content = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                    
        except Exception as e:
            self.logger.error(f"Error reading PDF {file_path}: {e}")
            
        return text_content.strip()
        
    async def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from Word documents"""
        try:
            doc = docx.Document(file_path)
            text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text_content.strip()
        except Exception as e:
            self.logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
            
    async def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Error reading TXT {file_path}: {e}")
            return ""
            
    async def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                return soup.get_text()
        except Exception as e:
            self.logger.error(f"Error reading HTML {file_path}: {e}")
            return ""
            
    async def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of the document"""
        try:
            # Split text into chunks for analysis
            chunks = self._split_text_into_chunks(text, max_length=512)
            
            sentiments = []
            
            if self.sentiment_pipeline:
                # Use FinBERT for financial sentiment
                for chunk in chunks[:10]:  # Limit to first 10 chunks for performance
                    try:
                        result = self.sentiment_pipeline(chunk)[0]
                        sentiments.append({
                            'label': result['label'],
                            'score': result['score']
                        })
                    except Exception as e:
                        self.logger.warning(f"FinBERT analysis failed for chunk: {e}")
                        continue
            else:
                # Fallback to TextBlob
                blob = TextBlob(text[:5000])  # Limit text length
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    label = "positive"
                elif polarity < -0.1:
                    label = "negative"
                else:
                    label = "neutral"
                    
                sentiments.append({
                    'label': label,
                    'score': abs(polarity)
                })
                
            # Aggregate sentiment results
            if sentiments:
                positive_count = sum(1 for s in sentiments if s['label'].lower() == 'positive')
                negative_count = sum(1 for s in sentiments if s['label'].lower() == 'negative')
                neutral_count = len(sentiments) - positive_count - negative_count
                
                avg_confidence = sum(s['score'] for s in sentiments) / len(sentiments)
                
                if positive_count > negative_count:
                    overall_sentiment = 'positive'
                elif negative_count > positive_count:
                    overall_sentiment = 'negative'
                else:
                    overall_sentiment = 'neutral'
                    
                return {
                    'overall_sentiment': overall_sentiment,
                    'confidence': avg_confidence,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count,
                    'detailed_results': sentiments[:5]  # Save top 5 results
                }
            else:
                return {
                    'overall_sentiment': 'neutral',
                    'confidence': 0.5,
                    'error': 'Could not analyze sentiment'
                }
                
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {'error': str(e)}
            
    async def _extract_key_information(self, text: str, document_type: str) -> Dict:
        """Extract key information based on document type"""
        key_info = {}
        
        try:
            # Common financial keywords
            financial_keywords = [
                'revenue', 'profit', 'earnings', 'dividend', 'growth', 'margin',
                'cash flow', 'debt', 'equity', 'assets', 'liabilities', 'return on equity',
                'ebitda', 'operating income', 'net income', 'total assets'
            ]
            
            # Extract mentions of financial terms
            financial_mentions = {}
            text_lower = text.lower()
            
            for keyword in financial_keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    financial_mentions[keyword] = count
                    
            key_info['financial_term_frequency'] = financial_mentions
            
            # Extract numerical values with financial context
            financial_numbers = self._extract_financial_numbers(text)
            key_info['financial_figures'] = financial_numbers
            
            # Document type specific extraction
            if document_type == 'annual_report':
                key_info.update(await self._extract_annual_report_info(text))
            elif document_type == 'asx_announcement':
                key_info.update(await self._extract_announcement_info(text))
            elif document_type == 'investor_presentation':
                key_info.update(await self._extract_presentation_info(text))
                
        except Exception as e:
            self.logger.error(f"Error extracting key information: {e}")
            key_info['error'] = str(e)
            
        return key_info
        
    def _extract_financial_numbers(self, text: str) -> List[Dict]:
        """Extract financial numbers and their context"""
        financial_numbers = []
        
        # Regex patterns for financial figures
        patterns = [
            r'\$([0-9,]+(?:\.[0-9]+)?)\s*(million|billion|m|bn|k)?',
            r'([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|AUD|USD|\$)',
            r'([0-9,]+(?:\.[0-9]+)?)%',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                financial_numbers.append({
                    'value': match.group(0),
                    'context': context,
                    'position': match.start()
                })
                
        return financial_numbers[:20]  # Limit to first 20 matches
        
    async def _extract_financial_metrics(self, text: str) -> Dict:
        """Extract specific financial metrics from text"""
        metrics = {}
        
        try:
            # Common financial ratios and metrics
            metric_patterns = {
                'roe': r'return on equity.*?([0-9.]+%)',
                'roa': r'return on assets.*?([0-9.]+%)',
                'debt_ratio': r'debt.*?ratio.*?([0-9.]+%?)',
                'profit_margin': r'profit margin.*?([0-9.]+%)',
                'dividend_yield': r'dividend yield.*?([0-9.]+%)',
                'eps': r'earnings per share.*?\$?([0-9.]+)',
                'pe_ratio': r'p/e ratio.*?([0-9.]+)',
            }
            
            text_lower = text.lower()
            
            for metric_name, pattern in metric_patterns.items():
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    metrics[metric_name] = matches[0]
                    
        except Exception as e:
            self.logger.error(f"Error extracting financial metrics: {e}")
            
        return metrics
        
    async def _analyze_risks(self, text: str) -> Dict:
        """Analyze risk factors mentioned in the document"""
        risk_analysis = {}
        
        try:
            # Risk-related keywords
            risk_keywords = [
                'risk', 'uncertainty', 'volatility', 'challenge', 'threat',
                'competition', 'regulation', 'market conditions', 'economic downturn',
                'credit risk', 'operational risk', 'liquidity risk'
            ]
            
            risk_mentions = {}
            text_lower = text.lower()
            
            for keyword in risk_keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    risk_mentions[keyword] = count
                    
            # Find risk sections
            risk_sections = []
            risk_section_indicators = ['risk factors', 'risks', 'risk management', 'principal risks']
            
            for indicator in risk_section_indicators:
                start = text_lower.find(indicator)
                if start != -1:
                    # Extract section (next 1000 characters)
                    section = text[start:start+1000]
                    risk_sections.append(section)
                    
            risk_analysis['risk_mentions'] = risk_mentions
            risk_analysis['risk_sections'] = risk_sections
            
            # Calculate risk score based on frequency
            total_risk_mentions = sum(risk_mentions.values())
            text_length = len(text.split())
            
            if text_length > 0:
                risk_score = min(1.0, total_risk_mentions / text_length * 100)
            else:
                risk_score = 0.0
                
            risk_analysis['risk_score'] = risk_score
            
        except Exception as e:
            self.logger.error(f"Error analyzing risks: {e}")
            
        return risk_analysis
        
    async def _extract_business_insights(self, text: str) -> Dict:
        """Extract business insights and strategic information"""
        insights = {}
        
        try:
            # Business insight keywords
            insight_keywords = [
                'strategy', 'outlook', 'guidance', 'forecast', 'expansion',
                'investment', 'innovation', 'market share', 'competitive advantage',
                'new products', 'acquisitions', 'partnerships'
            ]
            
            insight_mentions = {}
            text_lower = text.lower()
            
            for keyword in insight_keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    insight_mentions[keyword] = count
                    
            insights['strategic_mentions'] = insight_mentions
            
            # Extract future-looking statements
            future_indicators = ['expect', 'anticipate', 'forecast', 'project', 'outlook', 'guidance']
            future_statements = []
            
            sentences = text.split('.')
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in future_indicators):
                    future_statements.append(sentence.strip())
                    
            insights['forward_looking_statements'] = future_statements[:10]
            
        except Exception as e:
            self.logger.error(f"Error extracting business insights: {e}")
            
        return insights
        
    async def _generate_summary(self, text: str) -> Dict:
        """Generate a summary of the document"""
        try:
            # Limit text length for processing
            max_length = 10000
            if len(text) > max_length:
                text = text[:max_length]
                
            if self.summarizer and len(text) > 100:
                # Use transformer-based summarization
                try:
                    summary_result = self.summarizer(
                        text,
                        max_length=self.config['analysis']['max_summary_length'],
                        min_length=100,
                        do_sample=False
                    )
                    
                    return {
                        'summary': summary_result[0]['summary_text'],
                        'method': 'transformer',
                        'confidence': 0.8
                    }
                except Exception as e:
                    self.logger.warning(f"Transformer summarization failed: {e}")
            
            # Fallback: extractive summarization
            sentences = text.split('.')
            # Simple extractive summary - take first and last few sentences
            if len(sentences) > 5:
                summary_sentences = sentences[:3] + sentences[-2:]
                summary_text = '. '.join(summary_sentences).strip()
            else:
                summary_text = text[:500] + "..." if len(text) > 500 else text
                
            return {
                'summary': summary_text,
                'method': 'extractive',
                'confidence': 0.6
            }
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {
                'summary': text[:300] + "..." if len(text) > 300 else text,
                'method': 'truncated',
                'confidence': 0.3
            }
            
    def _split_text_into_chunks(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for space
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    async def _extract_annual_report_info(self, text: str) -> Dict:
        """Extract specific information from annual reports"""
        return {
            'report_type': 'annual_report',
            'sections_detected': self._detect_annual_report_sections(text)
        }
        
    async def _extract_announcement_info(self, text: str) -> Dict:
        """Extract specific information from ASX announcements"""
        return {
            'announcement_type': 'asx_announcement',
            'urgency_indicators': self._detect_urgency_indicators(text)
        }
        
    async def _extract_presentation_info(self, text: str) -> Dict:
        """Extract specific information from investor presentations"""
        return {
            'presentation_type': 'investor_presentation',
            'key_slides_detected': self._detect_key_presentation_content(text)
        }
        
    def _detect_annual_report_sections(self, text: str) -> List[str]:
        """Detect common annual report sections"""
        sections = []
        section_indicators = [
            'executive summary', 'financial highlights', 'operating review',
            'financial statements', 'notes to financial statements', 'risk factors',
            'corporate governance', 'remuneration report'
        ]
        
        text_lower = text.lower()
        for section in section_indicators:
            if section in text_lower:
                sections.append(section)
                
        return sections
        
    def _detect_urgency_indicators(self, text: str) -> List[str]:
        """Detect urgency indicators in announcements"""
        urgency_indicators = []
        urgent_keywords = ['immediate', 'urgent', 'halt', 'suspension', 'trading halt']
        
        text_lower = text.lower()
        for keyword in urgent_keywords:
            if keyword in text_lower:
                urgency_indicators.append(keyword)
                
        return urgency_indicators
        
    def _detect_key_presentation_content(self, text: str) -> List[str]:
        """Detect key presentation content"""
        key_content = []
        presentation_keywords = ['outlook', 'strategy', 'results', 'guidance', 'market update']
        
        text_lower = text.lower()
        for keyword in presentation_keywords:
            if keyword in text_lower:
                key_content.append(keyword)
                
        return key_content
        
    def _calculate_overall_confidence(self, analysis_results: Dict) -> float:
        """Calculate overall confidence score for the analysis"""
        confidences = []
        
        for analysis_type, results in analysis_results.items():
            if isinstance(results, dict) and 'confidence' in results:
                confidences.append(results['confidence'])
                
        return sum(confidences) / len(confidences) if confidences else 0.5
        
    async def _save_analysis_results(self, document_id: int, analysis_results: Dict, confidence: float):
        """Save analysis results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for analysis_type, results in analysis_results.items():
            cursor.execute('''
            INSERT INTO document_analysis (
                document_id, analysis_type, analysis_result, confidence_score, model_version
            ) VALUES (?, ?, ?, ?, ?)
            ''', (
                document_id, analysis_type, json.dumps(results), 
                confidence, self.config['prediction']['model_version']
            ))
            
        conn.commit()
        conn.close()
        
    def _get_document_info(self, document_id: int) -> Optional[Dict]:
        """Get document information from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT symbol, document_type, title, local_path, status
        FROM documents WHERE id = ?
        ''', (document_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'symbol': result[0],
                'document_type': result[1],
                'title': result[2],
                'local_path': result[3],
                'status': result[4]
            }
        return None
        
    def _is_already_analyzed(self, document_id: int) -> bool:
        """Check if document has already been analyzed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT COUNT(*) FROM document_analysis WHERE document_id = ?
        ''', (document_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
        
    def _update_document_analysis_status(self, document_id: int):
        """Update document analysis status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE documents 
        SET status = 'analyzed', last_analyzed = CURRENT_TIMESTAMP 
        WHERE id = ?
        ''', (document_id,))
        
        conn.commit()
        conn.close()
        
    async def get_document_analysis(self, document_id: int) -> Dict:
        """Get all analysis results for a document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT analysis_type, analysis_result, confidence_score, created_date
        FROM document_analysis 
        WHERE document_id = ?
        ORDER BY created_date DESC
        ''', (document_id,))
        
        results = {}
        for row in cursor.fetchall():
            analysis_type = row[0]
            analysis_result = json.loads(row[1])
            confidence_score = row[2]
            created_date = row[3]
            
            results[analysis_type] = {
                'result': analysis_result,
                'confidence': confidence_score,
                'analyzed_date': created_date
            }
            
        conn.close()
        return results
        
    async def batch_analyze_documents(self, symbol: str) -> Dict:
        """Analyze all documents for a symbol"""
        self.logger.info(f"🔄 Starting batch analysis for {symbol}")
        
        # Get all unanalyzed documents for the symbol
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id FROM documents 
        WHERE symbol = ? AND status = 'downloaded'
        ORDER BY download_date DESC
        ''', (symbol,))
        
        document_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        results = {
            'symbol': symbol,
            'total_documents': len(document_ids),
            'analyzed': 0,
            'failed': 0,
            'results': []
        }
        
        for doc_id in document_ids:
            try:
                analysis_result = await self.analyze_document(doc_id)
                if analysis_result['success']:
                    results['analyzed'] += 1
                else:
                    results['failed'] += 1
                    
                results['results'].append(analysis_result)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error analyzing document {doc_id}: {e}")
                results['failed'] += 1
                
        self.logger.info(f"✅ Batch analysis completed for {symbol}: "
                        f"{results['analyzed']} analyzed, {results['failed']} failed")
                        
        return results