#pragma once

#include "module.h"

#include <sstream>
#include <unistd.h>
#include <cmath>

class HelloModule : public Module
{
	public:
		HelloModule()
		{
			srand( time(NULL) );
		}

		virtual ~HelloModule()
		{

		}

		void tick()
		{
			float number = 1.5;
			for( size_t k = 0; k < 1000000; ++k )
			{
				number*=number;
				number = pow(number, number);
			}
			std::stringstream sstr;
			sstr << "popo: " << rand();;
			Message msg(sstr.str());
			emit( msg );
			// std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}

	private:

	protected:


};
