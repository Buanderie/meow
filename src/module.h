#pragma once

//
#include "message.h"

// STL
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>
#include <vector>
#include <queue>

#define QUEUE_MAX_SIZE 128

class App;
class Module
{
	friend class App;
	public:
		
		typedef enum
		{
			MODULE_STOPPED=0,
			MODULE_RUNNING,
			MODULE_WAITING_STOP
		} ModuleState;

		Module()
		:_state(MODULE_STOPPED), _parent(0)
		{
		}

		virtual ~Module(){

		}

		void start()
		{
			_thread = new std::thread( &Module::run, this );
			setState( MODULE_RUNNING );
		}

		void stop()
		{
			setState( MODULE_WAITING_STOP );
		}

		void waitForTermination()
		{

		}

		ModuleState getState()
		{
			std::lock_guard<std::mutex> lock(_moduleMtx);
			return _state;	
		}

		void setState( ModuleState state )
		{
			std::lock_guard<std::mutex> lock(_moduleMtx);
			_state = state;
		}

		void addRecipient( const std::string& recipientName )
		{
			std::lock_guard<std::mutex> lock(_moduleMtx);
			_recipients.push_back( recipientName );
		}

	private:
		std::thread * 				_thread;
		std::mutex	  				_moduleMtx;
		
		ModuleState					_state;

		App*						_parent;
		std::vector< std::string > 	_recipients;
		std::queue< Message >		_msgQueue;
		std::mutex					_queueMtx;
		std::condition_variable 	_queueReadCond;
		std::condition_variable 	_queueWriteCond;

		void run()
		{
			while( 	getState() != MODULE_WAITING_STOP )
			{
				tick();
			}	
			setState( MODULE_STOPPED );	
		}

		void setParentApp( App* app )
		{
			_parent = app;
		}

		bool enqueueMessage( const Message& msg, bool isBlocking=true )
		{
			std::unique_lock<std::mutex> lock(_queueMtx);
			if( isBlocking )
			{
				while(true)
				{
					if( _msgQueue.size() < QUEUE_MAX_SIZE )
					{
						_msgQueue.push( msg );
						_queueReadCond.notify_all();
						break;
					}	
					else
					{
						_queueWriteCond.wait( lock );
					}
				}
				return true;
			}
			else
			{
				if( _msgQueue.size() < QUEUE_MAX_SIZE )
				{
					_msgQueue.push( msg );
				    _queueReadCond.notify_all();
					return true;
				}
				else
				{
					return false;
				}
			}
		}

	protected:
		virtual void tick()=0;
		void emit( Message& msg );
		bool receive( Message& msg, bool isBlocking=true )
		{
			std::unique_lock<std::mutex> lock(_queueMtx);
			if( isBlocking )
			{
				while(true)
				{
					if( _msgQueue.size() > 0 )
					{
						msg = _msgQueue.front();
						_msgQueue.pop();
						_queueWriteCond.notify_all();
						break;
					}
					else
					{
						_queueReadCond.wait(lock);
					}
				}
				return true;
			}
			else
			{
				if( _msgQueue.size() > 0 )
				{
					msg = _msgQueue.front();
					_msgQueue.pop();
					_queueWriteCond.notify_all();
					return true;
				}
				else
				{
					return false;
				}
			}
		}

};

