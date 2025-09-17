"""
Document Downloader - Enhanced Stock Market Tracker Local
Downloads and manages financial documents for comprehensive stock analysis
"""

import os
import asyncio
import aiohttp
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
import json
import logging
from dataclasses import dataclass
from bs4 import BeautifulSoup
import requests
import time

@dataclass
class DocumentInfo:
    title: str
    url: str
    document_type: str
    published_date: Optional[datetime] = None
    file_size: Optional[int] = None
    content_type: Optional[str] = None

class DocumentDownloader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.db_path = self.config['database']['path']
        self.documents_dir = Path(self.config['document_management']['download_dir'])
        self.max_file_size = self.config['document_management']['max_file_size_mb'] * 1024 * 1024
        self.allowed_extensions = set(self.config['document_management']['allowed_extensions'])
        
        # Create documents directory structure
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        for doc_type in ['annual_reports', 'asx_announcements', 'investor_presentations', 
                        'financial_statements', 'news_articles', 'research_reports']:
            (self.documents_dir / doc_type).mkdir(exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
        
    async def download_all_documents(self, symbol: str, force_refresh: bool = False) -> Dict:
        """Download all available documents for a stock symbol"""
        self.logger.info(f"📥 Starting comprehensive document download for {symbol}")
        
        results = {
            'symbol': symbol,
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'documents': []
        }
        
        try:
            # Get available documents from multiple sources
            document_sources = await self._discover_all_documents(symbol)
            
            # Download each document
            for doc_info in document_sources:
                try:
                    result = await self._download_document(symbol, doc_info, force_refresh)
                    if result['success']:
                        results['downloaded'] += 1
                        results['documents'].append(result)
                    else:
                        results['skipped' if result.get('skipped') else 'failed'] += 1
                        
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error downloading {doc_info.title}: {e}")
                    results['failed'] += 1
                    
            self.logger.info(f"✅ Document download completed for {symbol}: "
                           f"{results['downloaded']} downloaded, {results['skipped']} skipped, "
                           f"{results['failed']} failed")
                           
        except Exception as e:
            self.logger.error(f"Error in document discovery for {symbol}: {e}")
            
        return results
        
    async def _discover_all_documents(self, symbol: str) -> List[DocumentInfo]:
        """Discover all available documents for a stock from multiple sources"""
        all_documents = []
        
        # ASX Announcements
        asx_docs = await self._get_asx_announcements(symbol)
        all_documents.extend(asx_docs)
        
        # Annual Reports (company website)
        annual_reports = await self._get_annual_reports(symbol)
        all_documents.extend(annual_reports)
        
        # Investor Presentations
        presentations = await self._get_investor_presentations(symbol)
        all_documents.extend(presentations)
        
        # Financial Statements
        financials = await self._get_financial_statements(symbol)
        all_documents.extend(financials)
        
        # Research Reports (if available)
        research = await self._get_research_reports(symbol)
        all_documents.extend(research)
        
        self.logger.info(f"📋 Discovered {len(all_documents)} documents for {symbol}")
        return all_documents
        
    async def _get_asx_announcements(self, symbol: str) -> List[DocumentInfo]:
        """Get ASX announcements for a stock"""
        documents = []
        
        try:
            # Remove .AX suffix for ASX queries
            asx_code = symbol.replace('.AX', '')
            
            # ASX announcements URL
            url = f"https://www.asx.com.au/asx/1/company/{asx_code}/announcements"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for announcement in data.get('data', []):
                            doc_info = DocumentInfo(
                                title=announcement.get('header', 'ASX Announcement'),
                                url=f"https://www.asx.com.au/asxpdf/{announcement.get('document_release_date')[:4]}/{announcement.get('url')}",
                                document_type='asx_announcement',
                                published_date=datetime.strptime(announcement.get('document_release_date'), '%Y-%m-%dT%H:%M:%S%z') if announcement.get('document_release_date') else None
                            )
                            documents.append(doc_info)
                            
        except Exception as e:
            self.logger.error(f"Error fetching ASX announcements for {symbol}: {e}")
            
        return documents[:20]  # Limit to most recent 20
        
    async def _get_annual_reports(self, symbol: str) -> List[DocumentInfo]:
        """Get annual reports from company website"""
        documents = []
        
        try:
            # Company website mapping (extend this based on your needs)
            company_websites = {
                'CBA.AX': 'https://www.commbank.com.au/about-us/investors/annual-reports.html',
                'WBC.AX': 'https://www.westpac.com.au/about-westpac/investor-centre/annual-reports/',
                'ANZ.AX': 'https://www.anz.com/shareholder/centre/reporting/annual-report/',
                'NAB.AX': 'https://www.nab.com.au/about-us/shareholder-centre/annual-reports'
            }
            
            if symbol not in company_websites:
                return documents
                
            url = company_websites[symbol]
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find PDF links (customize selectors based on website structure)
                        pdf_links = soup.find_all('a', href=lambda href: href and '.pdf' in href.lower())
                        
                        for link in pdf_links:
                            href = link.get('href')
                            if not href.startswith('http'):
                                href = urljoin(url, href)
                                
                            title = link.text.strip() or link.get('title', 'Annual Report')
                            
                            # Filter for annual report keywords
                            if any(keyword in title.lower() for keyword in ['annual report', 'annual review', 'financial report']):
                                doc_info = DocumentInfo(
                                    title=title,
                                    url=href,
                                    document_type='annual_report'
                                )
                                documents.append(doc_info)
                                
        except Exception as e:
            self.logger.error(f"Error fetching annual reports for {symbol}: {e}")
            
        return documents[:10]  # Limit to most recent 10
        
    async def _get_investor_presentations(self, symbol: str) -> List[DocumentInfo]:
        """Get investor presentations"""
        documents = []
        
        try:
            # Similar approach to annual reports but for presentations
            company_presentation_urls = {
                'CBA.AX': 'https://www.commbank.com.au/about-us/investors/presentations.html',
                'WBC.AX': 'https://www.westpac.com.au/about-westpac/investor-centre/presentations/',
                # Add more as needed
            }
            
            if symbol not in company_presentation_urls:
                return documents
                
            url = company_presentation_urls[symbol]
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        pdf_links = soup.find_all('a', href=lambda href: href and '.pdf' in href.lower())
                        
                        for link in pdf_links:
                            href = link.get('href')
                            if not href.startswith('http'):
                                href = urljoin(url, href)
                                
                            title = link.text.strip() or 'Investor Presentation'
                            
                            if any(keyword in title.lower() for keyword in ['presentation', 'briefing', 'update', 'results']):
                                doc_info = DocumentInfo(
                                    title=title,
                                    url=href,
                                    document_type='investor_presentation'
                                )
                                documents.append(doc_info)
                                
        except Exception as e:
            self.logger.error(f"Error fetching presentations for {symbol}: {e}")
            
        return documents[:15]  # Limit to most recent 15
        
    async def _get_financial_statements(self, symbol: str) -> List[DocumentInfo]:
        """Get quarterly and half-yearly financial statements"""
        documents = []
        
        try:
            # This would typically come from ASX or company investor relations
            # Implementation similar to above methods
            pass
            
        except Exception as e:
            self.logger.error(f"Error fetching financial statements for {symbol}: {e}")
            
        return documents
        
    async def _get_research_reports(self, symbol: str) -> List[DocumentInfo]:
        """Get research reports from available sources"""
        documents = []
        
        try:
            # This would integrate with research providers if available
            # Could include broker reports, analyst notes, etc.
            pass
            
        except Exception as e:
            self.logger.error(f"Error fetching research reports for {symbol}: {e}")
            
        return documents
        
    async def _download_document(self, symbol: str, doc_info: DocumentInfo, force_refresh: bool = False) -> Dict:
        """Download a single document"""
        try:
            # Check if already exists
            existing_doc = self._get_existing_document(doc_info.url)
            if existing_doc and not force_refresh:
                return {
                    'success': True,
                    'skipped': True,
                    'message': 'Document already exists',
                    'local_path': existing_doc['local_path']
                }
                
            # Create filename
            filename = self._generate_filename(symbol, doc_info)
            local_path = self.documents_dir / doc_info.document_type / filename
            
            # Download file
            async with aiohttp.ClientSession() as session:
                async with session.get(doc_info.url) as response:
                    if response.status != 200:
                        return {
                            'success': False,
                            'message': f'HTTP {response.status}'
                        }
                        
                    # Check file size
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > self.max_file_size:
                        return {
                            'success': False,
                            'message': f'File too large: {content_length} bytes'
                        }
                        
                    # Download content
                    content = await response.read()
                    
                    # Save file
                    with open(local_path, 'wb') as f:
                        f.write(content)
                        
            # Calculate content hash
            content_hash = hashlib.md5(content).hexdigest()
            
            # Save to database
            doc_id = self._save_document_to_db(symbol, doc_info, str(local_path), len(content), content_hash)
            
            self.logger.info(f"📄 Downloaded: {doc_info.title}")
            
            return {
                'success': True,
                'document_id': doc_id,
                'local_path': str(local_path),
                'title': doc_info.title,
                'size': len(content)
            }
            
        except Exception as e:
            self.logger.error(f"Error downloading {doc_info.title}: {e}")
            return {
                'success': False,
                'message': str(e)
            }
            
    def _get_existing_document(self, url: str) -> Optional[Dict]:
        """Check if document already exists in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, local_path, title FROM documents WHERE url = ?
        ''', (url,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'local_path': result[1],
                'title': result[2]
            }
        return None
        
    def _generate_filename(self, symbol: str, doc_info: DocumentInfo) -> str:
        """Generate a safe filename for the document"""
        # Clean title
        safe_title = "".join(c for c in doc_info.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title[:50]  # Limit length
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Determine extension
        extension = '.pdf'  # Default
        if doc_info.url:
            parsed_url = urlparse(doc_info.url)
            if parsed_url.path:
                url_ext = os.path.splitext(parsed_url.path)[1].lower()
                if url_ext in self.allowed_extensions:
                    extension = url_ext
                    
        return f"{symbol}_{safe_title}_{timestamp}{extension}"
        
    def _save_document_to_db(self, symbol: str, doc_info: DocumentInfo, local_path: str, 
                           file_size: int, content_hash: str) -> int:
        """Save document information to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO documents (
            symbol, document_type, title, url, local_path, 
            file_size, content_hash, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, doc_info.document_type, doc_info.title, doc_info.url,
            local_path, file_size, content_hash, 'downloaded'
        ))
        
        document_id = cursor.lastrowid
        
        # Update stock document count
        cursor.execute('''
        INSERT OR REPLACE INTO stocks (symbol, document_count) 
        VALUES (?, (SELECT COUNT(*) FROM documents WHERE symbol = ?))
        ''', (symbol, symbol))
        
        conn.commit()
        conn.close()
        
        return document_id
        
    def get_downloaded_documents(self, symbol: str) -> List[Dict]:
        """Get list of all downloaded documents for a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, document_type, title, local_path, file_size, 
               download_date, last_analyzed, status
        FROM documents 
        WHERE symbol = ? 
        ORDER BY download_date DESC
        ''', (symbol,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'document_type': row[1],
                'title': row[2],
                'local_path': row[3],
                'file_size': row[4],
                'download_date': row[5],
                'last_analyzed': row[6],
                'status': row[7]
            })
            
        conn.close()
        return results
        
    async def schedule_download(self, symbol: str) -> None:
        """Schedule regular document downloads for a symbol"""
        self.logger.info(f"📅 Scheduling regular downloads for {symbol}")
        
        while True:
            try:
                await self.download_all_documents(symbol)
                
                # Wait 24 hours before next download
                await asyncio.sleep(24 * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in scheduled download for {symbol}: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour on error