#pragma once

// STL
#include <string>
#include <chrono>
#include <sstream>
#include <ctime>

class Module;
class Message
{
	friend class Module;
	public:
		Message( 	const std::string& content = "", 
					const std::string& topic = "" )
		:_content(content), _topic(topic)
		{

		}

		Message( const Message& other )
		{
			_content = other._content;
			_topic = other._topic;
			_sendingTime = other._sendingTime;
		}

		virtual ~Message()
		{

		}

		std::string getContent()
		{
			return _content;
		}

		std::string getTimeAsString()
		{
				std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(_sendingTime.time_since_epoch());
			std::stringstream sstr;
			// std::time_t ttp = std::chrono::high_resolution_clock::to_time_t(_sendingTime);
			// sstr << std::ctime(&ttp) << std::size_t fractional_seconds = ms.count() % 1000;
			std::size_t fractional_seconds = ms.count() % 1000;
			std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(ms);
			sstr << s.count() << "." << fractional_seconds;
			return sstr.str();
		}

	private:
		std::string _content;
		std::string _topic;
		std::chrono::high_resolution_clock::time_point _sendingTime;

		void setSendingTime( const std::chrono::high_resolution_clock::time_point t )
		{
			_sendingTime = t;	
		}

	protected:


};
