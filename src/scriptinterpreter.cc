
#include "scriptinterpreter.h"
#include "scriptedmodule.h"
#include "message.h"

#include <lua.hpp>
#include <LuaBridge.h>

using namespace std;

using namespace luabridge;

void ScriptInterpreter::exposeModuleFunctions()
{
    // Define Lua exports
    getGlobalNamespace((lua_State*)L)

    .beginClass<Message>("Message")
            //.addConstructor< void(*)() >()
            .addConstructor< void(*)(const Message&) >()
            .addConstructor< void(*)(const std::string&, const std::string&) >()
            .addFunction ("__", &Message::operator=)
            .addFunction("getContent", &Message::getContent )
            .addFunction("getTimeAsString", &Message::getTimeAsString )
            .endClass()
            
    .beginClass<ScriptedModule>("ScriptedModule")
            .addConstructor< void(*)(void) >()
            .addFunction("emit", &ScriptedModule::emit )
            .addFunction("receive", &ScriptedModule::scriptReceive )
            .endClass();

    // Export our module reference
    setGlobal((lua_State*)L, (ScriptedModule*)_parentModule, "module" );
    
	/*
	lualite::module{(lua_State*)L,
    lualite::class_<Message>("Message")
    	.constructor<std::string, std::string>("new")
      .constant("__classname", "Message")
      .constant("__b", true)
      .constant("__pi", 3.1459)
      .def<decltype(&Message::getContent), &Message::getContent>("getContent")
      .def<decltype(&Message::getContent), &Message::getTimeAsString>("getTimeAsString")
  };
  */
  
	// std::cout << "test test - " << LUA_VERSION_NUM <<  endl;
	// exit(0);
	/*
	_state["Message"].SetClass<Message, std::string, std::string>
									("getContent", 			&Message::getContent,
									 "getTimeAsString", &Message::getTimeAsString );
  								
	std::function<void(Message& msg)> temp_emit = [&](Message& msg){ this->_parentModule->emit(msg); };         							 
	_state["emit"] = temp_emit;
	
	std::function<bool(Message& msg, bool)> temp_receive = [&](Message& msg, bool isBlocking) -> bool { return this->_parentModule->receive(msg, isBlocking); };         							 
	_state["receive"] = temp_receive;
	
	std::function<void(std::string)> temp_addRecipient = [&](const std::string& moduleName){ this->_parentModule->addRecipient(moduleName); };         							 
	_state["addRecipient"] = temp_addRecipient;
	*/
	
}

void ScriptInterpreter::callreceive( Message& msg, bool isBlocking )
{
    _parentModule->receive( msg, isBlocking );
}
