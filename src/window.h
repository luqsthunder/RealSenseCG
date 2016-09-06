#ifndef REALSENSECG_WINDOW_H
#define REALSENSECG_WINDOW_H

#include <SDL_video.h>

namespace rscg
{

class Window
{
public:
  Window(unsigned w = 800, unsigned h = 600);
  ~Window();

  operator SDL_Window*() const;

protected:
  SDL_Window *_window;
  SDL_GLContext _context;
};

}
#endif //REALSENSECG_WINDOW_H
