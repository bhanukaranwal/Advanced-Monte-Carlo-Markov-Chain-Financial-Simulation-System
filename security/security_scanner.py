"""
Security scanning and vulnerability assessment
"""

import subprocess
import json
import logging
from typing import Dict, List, Any
import requests
import hashlib
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class SecurityScanner:
    """Comprehensive security scanning and vulnerability assessment"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.scan_results = {}
        
    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security linting"""
        logger.info("Running Bandit security scan...")
        
        try:
            result = subprocess.run([
                'bandit', '-r', str(self.project_root / 'src'),
                '-f', 'json', '-q'
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                scan_data = json.loads(result.stdout)
                return {
                    'status': 'success',
                    'issues': scan_data.get('results', []),
                    'total_issues': len(scan_data.get('results', [])),
                    'metrics': scan_data.get('metrics', {})
                }
            else:
                return {
                    'status': 'error',
                    'error': result.stderr,
                    'total_issues': 0
                }
                
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def run_safety_check(self) -> Dict[str, Any]:
        """Run Safety vulnerability check on dependencies"""
        logger.info("Running Safety dependency check...")
        
        try:
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                return {
                    'status': 'success',
                    'vulnerabilities': [],
                    'total_vulnerabilities': 0
                }
            else:
                try:
                    vulns = json.loads(result.stdout)
                    return {
                        'status': 'warning',
                        'vulnerabilities': vulns,
                        'total_vulnerabilities': len(vulns)
                    }
                except json.JSONDecodeError:
                    return {
                        'status': 'error',
                        'error': result.stderr
                    }
                    
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def check_secret_exposure(self) -> Dict[str, Any]:
        """Check for exposed secrets and credentials"""
        logger.info("Checking for exposed secrets...")
        
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
            r'secret[_-]?key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'aws[_-]?access[_-]?key',
            r'aws[_-]?secret[_-]?key'
        ]
        
        exposed_secrets = []
        
        for pattern in secret_patterns:
            try:
                result = subprocess.run([
                    'grep', '-r', '-i', '--exclude-dir=.git',
                    '--exclude-dir=venv', '--exclude-dir=__pycache__',
                    pattern, str(self.project_root)
                ], capture_output=True, text=True, check=False)
                
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line:
                            exposed_secrets.append({
                                'pattern': pattern,
                                'match': line,
                                'severity': 'high'
                            })
                            
            except Exception as e:
                logger.error(f"Secret scan error for pattern {pattern}: {e}")
                
        return {
            'status': 'success',
            'exposed_secrets': exposed_secrets,
            'total_secrets': len(exposed_secrets)
        }
        
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check file permissions for security issues"""
        logger.info("Checking file permissions...")
        
        security_issues = []
        
        # Check for world-writable files
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    stat = os.stat(filepath)
                    mode = stat.st_mode
                    
                    # Check for world-writable files
                    if mode & 0o002:
                        security_issues.append({
                            'file': filepath,
                            'issue': 'world_writable',
                            'permissions': oct(mode)[-3:],
                            'severity': 'medium'
                        })
                        
                    # Check for overly permissive executable files
                    if (mode & 0o111) and (mode & 0o077):
                        security_issues.append({
                            'file': filepath,
                            'issue': 'overly_permissive_executable',
                            'permissions': oct(mode)[-3:],
                            'severity': 'low'
                        })
                        
                except OSError:
                    continue
                    
        return {
            'status': 'success',
            'permission_issues': security_issues,
            'total_issues': len(security_issues)
        }
        
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        logger.info("Generating comprehensive security report...")
        
        # Run all security checks
        bandit_results = self.run_bandit_scan()
        safety_results = self.run_safety_check()
        secrets_results = self.check_secret_exposure()
        permissions_results = self.check_file_permissions()
        
        # Calculate overall security score
        total_issues = (
            bandit_results.get('total_issues', 0) +
            safety_results.get('total_vulnerabilities', 0) +
            secrets_results.get('total_secrets', 0) +
            permissions_results.get('total_issues', 0)
        )
        
        # Security score (100 is perfect, 0 is very poor)
        if total_issues == 0:
            security_score = 100
        elif total_issues < 5:
            security_score = 85
        elif total_issues < 10:
            security_score = 70
        elif total_issues < 20:
            security_score = 50
        else:
            security_score = 25
            
        report = {
            'timestamp': datetime.now().isoformat(),
            'security_score': security_score,
            'total_issues': total_issues,
            'scans': {
                'code_security': bandit_results,
                'dependency_vulnerabilities': safety_results,
                'secret_exposure': secrets_results,
                'file_permissions': permissions_results
            },
            'recommendations': self._generate_recommendations(
                bandit_results, safety_results, secrets_results, permissions_results
            )
        }
        
        # Save report
        report_file = self.project_root / 'security_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Security report saved to {report_file}")
        logger.info(f"Security Score: {security_score}/100")
        
        return report
        
    def _generate_recommendations(self, bandit, safety, secrets, permissions) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if bandit.get('total_issues', 0) > 0:
            recommendations.append("Address Bandit security issues in code")
            
        if safety.get('total_vulnerabilities', 0) > 0:
            recommendations.append("Update vulnerable dependencies identified by Safety")
            
        if secrets.get('total_secrets', 0) > 0:
            recommendations.append("Remove exposed secrets and use environment variables")
            
        if permissions.get('total_issues', 0) > 0:
            recommendations.append("Fix file permission security issues")
            
        if not recommendations:
            recommendations.append("Security scan passed! Continue regular security monitoring")
            
        return recommendations

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scanner = SecurityScanner(".")
    report = scanner.generate_security_report()
    print(f"Security Score: {report['security_score']}/100")
