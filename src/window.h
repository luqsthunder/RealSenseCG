#ifndef REALSENSECG_WINDOW_H
#define REALSENSECG_WINDOW_H

#include <SDL_video.h>

class Window
{
public:
  Window();
  ~Window();

  operator SDL_Window*() const;

protected:
  SDL_Window *window;
  SDL_GLContext context;
};


#endif //REALSENSECG_WINDOW_H
