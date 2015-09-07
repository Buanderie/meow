#pragma once

// 
#include "message.h"

//
#include <module.h>

// STL
#include <string>
#include <map>

class App
{
	public:
		App(){}
		virtual ~App(){}

		bool addModule( const std::string& name, Module* module )
		{
			auto p = _modules.insert( make_pair( name, module ) );
			if( !p.second )
			{
				return false;	
			}
			module->setParentApp( this );
			return true;
		}

		void start()
		{
			for( auto m : _modules )
			{
				m.second->start();
			}
		}

		void stop()
		{
			for( auto m : _modules )
			{
				m.second->stop();
			}
		}

		void emit( const Message& msg, const std::vector< std::string >& recipients )
		{
			for( auto r : recipients )
			{
				Module* m = _modules[r];
				m->enqueueMessage( msg );
			}	
		}

		Module* modules( const std::string& moduleName )
		{
			return _modules[ moduleName ];
		}

	private:
		std::map< std::string, Module* > _modules;

	protected:

};
