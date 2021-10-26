import sqlite3
from contextlib import closing
import multiprocessing 
import copy

class DataDaemon:
    def __init__(self,db_name='db/results.db',overwrite=False):
        self.db_name = db_name
        self.daemon = None
        self.manager = None
        self.queue = None
        
        with closing(sqlite3.connect(db_name)) as connection:
            with closing(connection.cursor()) as cursor:
                
                if overwrite:
                    cursor.execute('DROP TABLE IF EXISTS params')
                    cursor.execute('DROP TABLE IF EXISTS ground_truth_plots')
                    cursor.execute('DROP TABLE IF EXISTS result_plots')
                    connection.commit()
                    
                cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="params"')
                if cursor.fetchone() is None: # table exists
                    cursor.execute(
                        "CREATE TABLE params"
                        "("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "npts INTEGER, "
                        "noise REAL, "
                        "method TEXT, "
                        "affinity TEXT, "
                        "co_affinity TEXT, "
                        "gamma REAL, "
                        "degree REAL, "
                        "c0 REAL, "
                        "co_gamma REAL, "
                        "fms REAL, "
                        "ground_truth_plot_id INT, "
                        "result_plot_id INT"
                        ")"
                    )
                    cursor.execute(
                        "CREATE TABLE ground_truth_plots"
                        "("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "npts INTEGER, "
                        "plot BLOB"
                        ")"
                    )
                    cursor.execute(
                        "CREATE TABLE result_plots"
                        "("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "plot BLOB"
                        ")"
                    )
                    
    def add_ground_truth_plot(self,npts,blob):
        with closing(sqlite3.connect(self.db_name)) as connection:
            with closing(connection.cursor()) as cursor:
                cursor.execute(f"SELECT id FROM ground_truth_plots WHERE npts=={npts}")
                ground_truth_id = cursor.fetchone()
                
                if ground_truth_id:
                    print(f'--> Skipping adding ground_truth_plot with npts={npts} as it already exists...')
                else:
                    cursor.execute("INSERT INTO ground_truth_plots(npts,plot) VALUES (?,?)",(npts,blob))
            connection.commit()
                    
        
    def exists(self,param_values):
        param_names = ['npts', 'noise', 'method', 'affinity', 'co_affinity', 'gamma', 'degree', 'c0', 'co_gamma']
        if len(param_names)!=len(param_values):
            raise ValueError(f'Must pass {len(param_names)} values to exists: {param_names}')
            
        params = [f'{name}="{value}"' for name,value in zip(param_names,param_values)]
        param_str = ' AND '.join(params)
        with closing(sqlite3.connect(self.db_name)) as connection:
            with closing(connection.cursor()) as cursor:
                cursor.execute(f'SELECT COUNT(*) FROM params WHERE {param_str}')
                count = cursor.fetchone()[0]
        if count==1:
            exists = True
        elif count==0:
            exists = False
        else:
            raise ValueError(f'Something went wrong with sql query. count={count}')
        return exists
        
                
    def start(self,chunksize=1):
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue()
        
        if chunksize>1:
            self.daemon = multiprocessing.Process(target=DataDaemon._runloop_chunked, args=(self.queue,self.db_name,chunksize))
        else:
            self.daemon = multiprocessing.Process(target=DataDaemon._runloop, args=(self.queue,self.db_name))
        self.daemon.daemon = True
        self.daemon.start()        
        return self.queue
    
    @staticmethod
    def _runloop_chunked(queue,db_name,chunksize):
        # #create ground_truth_mapping
        # with closing(sqlite3.connect(db_name)) as connection:
        #     with closing(connection.cursor()) as cursor:
        #         cursor.execute(f"SELECT (id,npts) FROM ground_truth_plots")
        #         mapping = {i[1]:i[0] for k in cursor.fetchall()}
        
        chunked = []
        finished = False
        while (not finished):
            q_item = queue.get()
            if q_item is None:
                finished = True
            else:
                chunked.append(q_item)
                
            if (len(chunked)>=chunksize) or (finished and len(chunked)>0):
                with closing(sqlite3.connect(db_name)) as connection:
                    with closing(connection.cursor()) as cursor:
                        for params,result_plot in chunked:
                            cursor.execute("INSERT INTO result_plots(plot) VALUES (?)",(result_plot,))
                            result_id = copy.copy(cursor.lastrowid)
                            
                            #get ground_truth id based on
                            cursor.execute(f"SELECT id FROM ground_truth_plots WHERE npts=={params[0]}")
                            ground_truth_id = cursor.fetchone()[0]
                            
                            params_plots = list(params)
                            params_plots.append(result_id)
                            params_plots.append(ground_truth_id)
                            
                            qmarks = '?,'*len(params_plots)
                            cursor.execute("INSERT INTO params(npts, noise, method, affinity, co_affinity, gamma, degree, c0, co_gamma, fms, result_plot_id, ground_truth_plot_id) VALUES ("+qmarks[:-1]+")",params_plots)
                    
                    connection.commit()
                chunked = []
                
    @staticmethod
    def _runloop(queue,db_name):
        while True:
            q_item = queue.get()
            if q_item is None:
                break
            params,result_plot = q_item
                
            with closing(sqlite3.connect(db_name)) as connection:
                with closing(connection.cursor()) as cursor:
                    cursor.execute("INSERT INTO result_plots(plot) VALUES (?)",(result_plot,))
                    result_id = copy.copy(cursor.lastrowid)
                    
                    #get ground_truth id based on
                    cursor.execute(f"SELECT id FROM ground_truth_plots WHERE npts=={params[0]}")
                    ground_truth_id = cursor.fetchone()[0]
                    
                    params_plots = list(params)
                    params_plots.append(result_id)
                    params_plots.append(ground_truth_id)
                    
                    qmarks = '?,'*len(params_plots)
                    cursor.execute("INSERT INTO params(npts, noise, method, affinity, co_affinity, gamma, degree, c0, co_gamma, fms, result_plot_id, ground_truth_plot_id) VALUES ("+qmarks[:-1]+")",params_plots)
                    
                    
                connection.commit()
                
    def stop(self):
        self.queue.put(None)#write_daemon will stop at None
        self.daemon.join()#wait for write_daemon to hit None