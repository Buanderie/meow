#pragma once

//
#include "message.h"

#include <lua.hpp>

class ScriptedModule;
class ScriptInterpreter
{
	public:
		ScriptInterpreter( ScriptedModule* parentModule )
		:_parentModule(parentModule)
		{
			L = (void*)luaL_newstate();
  		luaL_openlibs((lua_State*)L);
			exposeModuleFunctions();
			fromFile("/home/buanderie/tick.lua");
		}
		
		virtual ~ScriptInterpreter()
		{
		
		}
		
		void fromString( const std::string& sourceCode )
		{
			// _state(sourceCode.c_str());
		}
		
		void fromFile( const std::string& filePath )
		{
			// _state.Load( filePath );
			if (luaL_dofile((lua_State*)L, filePath.c_str()))
    	{
				printf("Could not load file: %s\n", lua_tostring((lua_State*)L, -1));
    	}
		}

		void callTick()
		{
			// _state["tick"]();
			lua_getglobal((lua_State*)L, "tick");
		  if( lua_isfunction((lua_State*)L, lua_gettop((lua_State*)L)) )
		  {
		      lua_call((lua_State*)L, 0, 0);
		  }
		}

        void callreceive( Message& msg, bool isBlocking );
		
	private:
		// sel::State _state;
		void* L;
		
		void exposeModuleFunctions();
		
		ScriptedModule* _parentModule;
		
	protected:
	
};
