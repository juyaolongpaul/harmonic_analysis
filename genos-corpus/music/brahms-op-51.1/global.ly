\version "2.12.0"

global = {
  %% aqui as configurações globais (andamento, compasso, etc)
  \key c \minor
  \time 3/2
  \repeat volta 2 {
    s1*3/2*31
    \mark \default
    s1*3/2*26
    \mark \default
    s1*3/2*22
  }
  \alternative {
    {
      %% casa 1:
      s1*3/2*3
    }
    %% casa 2:
    {
      s1*3/2
    }
  }
  s1*3/2*3
  \bar "||"
  \key c \major
  s1*3/2*25
  \bar "||"
  \key e \major
  s1*3/2*21
  \bar "||"
  \key c \minor
  s1*3/2*40
  \mark \default
  s1*3/2*24
  \mark \default
  s1*3/2*30
  \bar "||"
  \time 2/2
  s1*2/2*37
  \bar "|."
}
