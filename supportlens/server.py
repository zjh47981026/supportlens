"""Single-user local web workspace with bounded jobs and server-owned evidence."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sqlite3
import threading
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from .store import Store, parse_tickets
from .drafting import draft_response

STATIC = Path(__file__).parent / 'static'
MAX_BODY = 3_000_000

class Application:
    def __init__(self, root, store=None, drafter=None):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        self.store=store or Store(root/'knowledge')
        self.drafter=drafter or draft_response
        self.db=sqlite3.connect(root/'workspace.sqlite',check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, status TEXT, label TEXT, result TEXT, error TEXT, created TEXT)')
        self.db.execute("UPDATE jobs SET status='failed',error='The process stopped before this operation finished. Please try again.' WHERE status IN ('queued','running')")
        self.db.commit()
        self.lock=threading.RLock()
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.busy=threading.BoundedSemaphore(1)
        self.token=secrets.token_urlsafe(32)
        self.active=None

    @staticmethod
    def identifier(value):
        try:
            result=str(UUID(value))
            if result!=value: raise ValueError()
            return result
        except (ValueError,TypeError,AttributeError):
            raise ValueError('Invalid operation ID.') from None

    def jobs(self):
        with self.lock:
            rows=self.db.execute('SELECT id,kind,status,label,error,created FROM jobs ORDER BY rowid DESC LIMIT 100').fetchall()
        return [dict(zip(('id','kind','status','label','error','created'),r)) for r in rows]

    def job(self, identifier):
        identifier=self.identifier(identifier)
        with self.lock:
            row=self.db.execute('SELECT kind,status,label,result,error,created FROM jobs WHERE id=?',(identifier,)).fetchone()
        if not row: raise KeyError('Operation not found.')
        return {'id':identifier,**dict(zip(('kind','status','label','result','error','created'),row)), 'result':json.loads(row[3]) if row[3] else None}

    def submit(self, kind, label, action):
        if not self.busy.acquire(blocking=False):
            raise ValueError('Another operation is working. Please wait until it finishes.')
        identifier=str(uuid4())
        try:
            with self.lock:
                if self.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]>=100:
                    raise ValueError('This workspace has reached 100 saved operations. Start the app with a new data directory.')
                self.db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?)',(identifier,kind,'queued',label,'','','%s'%datetime.now(timezone.utc).isoformat()))
                self.db.commit();self.active=identifier
            self.pool.submit(self._run,identifier,action)
            return identifier
        except Exception:
            self.busy.release();raise

    def _run(self, identifier, action):
        try:
            with self.lock:
                self.db.execute("UPDATE jobs SET status='running' WHERE id=?",(identifier,));self.db.commit()
            result=action()
            with self.lock:
                self.db.execute("UPDATE jobs SET status='complete',result=? WHERE id=?",(json.dumps(result,allow_nan=False),identifier));self.db.commit()
        except Exception as exc:
            # Expected validation/model errors remain visible; no fake successful fallback.
            error=str(exc)[:350] if isinstance(exc,(ValueError,KeyError)) else 'The operation could not finish. Check the local model and try again.'
            with self.lock:
                self.db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?",(error,identifier));self.db.commit()
        finally:
            with self.lock:self.active=None
            self.busy.release()

    def search(self, payload):
        query=payload.get('query');mode=payload.get('mode','hybrid');filters=payload.get('filters',{})
        if not isinstance(query,str) or not 3<=len(query.strip())<=1000:
            raise ValueError('Describe the issue in 3–1000 characters.')
        if mode not in ('keyword','vector','hybrid') or not isinstance(filters,dict) or set(filters)-{'product','category','status'} or any(not isinstance(v,str) or len(v)>80 for v in filters.values()):
            raise ValueError('Choose a search mode and valid filters.')
        limit=payload.get('limit',5)
        if type(limit) is not int or not 1<=limit<=10:raise ValueError('Choose one to ten results.')
        query=query.strip()
        def run():
            result=self.store.search(query,mode=mode,filters=filters,limit=limit)
            return {**result,'query':query,'filters':filters}
        return self.submit('search',query,run)

    def make_draft(self,payload):
        reference=self.job(payload.get('search_id'))
        mode=payload.get('mode','evidence')
        if reference['kind']!='search' or reference['status']!='complete' or mode not in ('evidence','ai'):
            raise ValueError('Select a completed search and a draft mode.')
        evidence=reference['result']
        selected=payload.get('ticket_ids')
        if selected is not None:
            if not isinstance(selected,list) or not 1<=len(selected)<=5 or any(not isinstance(i,str) for i in selected) or len(set(selected))!=len(selected):
                raise ValueError('Choose one to five distinct retrieved tickets.')
            allowed={item['ticket']['id']:item for item in evidence['results']}
            if any(i not in allowed for i in selected):
                raise ValueError('Draft sources must come from this saved search.')
            results=[allowed[i] for i in selected]
        else:
            results=evidence['results'][:1]
        def run():
            result=self.drafter(evidence['query'],results,mode=mode)
            return {**result,'query':evidence['query'],'search_id':reference['id'],'dataset':evidence['dataset'],'reviewed':False}
        return self.submit('draft',reference['label'],run)

    def approve(self,identifier,payload):
        text=payload.get('text')
        if not isinstance(text,str) or not 20<=len(text.strip())<=12000:
            raise ValueError('Review a response of 20–12,000 characters.')
        with self.lock:
            record=self.job(identifier)
            if record['kind']!='draft' or record['status']!='complete' or not record['result'].get('draft') or record['result'].get('reviewed'):
                raise ValueError('Choose an unreviewed, completed draft.')
            result=record['result'];result['edited']=text.strip()!=result['draft'].strip()
            result.update(reviewed=True,approved_text=text.strip(),reviewed_at=datetime.now(timezone.utc).isoformat())
            self.db.execute('UPDATE jobs SET result=? WHERE id=?',(json.dumps(result),identifier));self.db.commit()
        return self.job(identifier)

    def evaluate(self):
        from .evaluation import evaluate
        def run():
            import hashlib
            from .store import validate_tickets
            sample=json.loads((Path(__file__).parent/'fixtures'/'tickets.json').read_text())
            expected=hashlib.sha256(json.dumps(validate_tickets(sample),sort_keys=True).encode()).hexdigest()
            metadata=self.store.metadata()
            if not metadata.get('dataset') or metadata['dataset']['fingerprint']!=expected:
                raise ValueError('Load the synthetic sample before running its labeled evaluation.')
            return evaluate(self.store)
        return self.submit('evaluation','Retrieval evaluation',run)

    def close(self):
        self.pool.shutdown(wait=True);self.store.close();self.db.close()


def make_handler(app,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def host_ok(self):return self.headers.get('Host') in (f'127.0.0.1:{port}',f'localhost:{port}')
        def reply(self,code,value,kind='application/json'):
            data=json.dumps(value,allow_nan=False).encode() if kind=='application/json' else value
            self.send_response(code);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if not self.host_ok():return self.reply(403,{'error':'Invalid host.'})
            path=urlsplit(self.path).path
            try:
                if path=='/api/config':return self.reply(200,{'token':app.token})
                if path=='/api/state':return self.reply(200,{'knowledge':app.store.metadata(),'jobs':app.jobs(),'active':app.active})
                if path.startswith('/api/jobs/'):return self.reply(200,app.job(path.rsplit('/',1)[1]))
                files={'/':('index.html','text/html'),'/app.js':('app.js','text/javascript'),'/style.css':('style.css','text/css')}
                if path not in files:return self.reply(404,{'error':'Not found.'})
                name,kind=files[path];self.reply(200,(STATIC/name).read_bytes(),kind)
            except (KeyError,ValueError):self.reply(404,{'error':'Operation not found.'})
        def do_POST(self):
            if not self.host_ok() or self.headers.get('Origin') not in (None,f'http://127.0.0.1:{port}',f'http://localhost:{port}') or not secrets.compare_digest(self.headers.get('X-App-Token',''),app.token):
                return self.reply(403,{'error':'Refresh this local page before making changes.'})
            try:
                self.connection.settimeout(5)
                length=int(self.headers.get('Content-Length','0'))
                if self.headers.get('Transfer-Encoding') or not 1<=length<=MAX_BODY or self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    raise ValueError('Expected a bounded JSON request.')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):raise ValueError('Expected a JSON object.')
                path=urlsplit(self.path).path
                if path=='/api/sample':identifier=app.submit('import','Synthetic support tickets',app.store.load_sample)
                elif path=='/api/import':
                    name=payload.get('name');content=payload.get('content');kind=payload.get('kind')
                    if not isinstance(name,str) or not 1<=len(name.strip())<=80 or not isinstance(content,str) or len(content.encode())>2_000_000 or kind not in ('csv','json'):
                        raise ValueError('Import a named JSON or CSV file up to 2 MB.')
                    tickets=parse_tickets(content,kind=kind)
                    identifier=app.submit('import',name.strip(),lambda:app.store.import_tickets(name.strip(),tickets))
                elif path=='/api/search':identifier=app.search(payload)
                elif path=='/api/draft':identifier=app.make_draft(payload)
                elif path=='/api/evaluate':identifier=app.evaluate()
                elif path.startswith('/api/review/'):
                    return self.reply(200,app.approve(path.rsplit('/',1)[1],payload))
                else:return self.reply(404,{'error':'Not found.'})
                self.reply(202,{'id':identifier})
            except (ValueError,KeyError,TimeoutError,RecursionError) as exc:self.reply(400,{'error':str(exc)[:350]})
    return Handler


def main():
    parser=argparse.ArgumentParser(description='Run SupportLens locally')
    parser.add_argument('--port',type=int,default=8768);parser.add_argument('--data-dir',default='.runtime')
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('Choose a port between 1024 and 65535.')
    app=Application(args.data_dir)
    try:server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(app,args.port))
    except Exception:app.close();raise
    server.daemon_threads=True
    print(f'SupportLens: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();app.close()

if __name__=='__main__':main()
