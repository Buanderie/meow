
#include "module.h"
#include "app.h"

void Module::emit( Message& msg )
{
	if( _parent )
	{
		msg.setSendingTime( std::chrono::high_resolution_clock::now() );
		_parent->emit( msg, _recipients );
	}
	else
	{
		// throw
	}
}

