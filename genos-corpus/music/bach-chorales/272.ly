\version "2.10.33"

\header {
  title = "272 - Ich dank' dir, lieber Herre"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key bes \major
}

soprano = \relative c'' {
  \partial 4 
  bes4 
  bes bes \repeat volta 2 {
    bes c 
    aes g f c' 
    d c bes a8 g 
    a4 g f
  }
  \alternative {
    {
      bes bes bes
    } 
    {
      \partial 4 
      f'
    }
  }
  ees d c bes 
  bes8 c d4 c d 
  ees d c b 
  c2. f,4 
  bes c d ees 
  f ees8 d c4 ees 
  d c f bes,8 c 
  d4 c bes
}


alto = \relative c' {
  \partial 4 
  f4 
  g aes
  \repeat volta 2 {
    g g 
    g8 f4 e8 f4 f 
    f e f c4~
    c8 f4 e8 c4
  }
  \alternative {
    {
      f g f
    }
    {
      \partial 4 
      bes8 a
    }
  } 
  g4 f8 g a4 bes8 aes 
  g4 f f f 
  ees aes8 g4 f8 g4 
  g2. c,4 
  f g8 a bes4 a8 g 
  f4 g a bes 
  aes g f f8 g 
  f ees4 d8 d4
}


tenor = \relative c' {
  \partial 4 
  d4 
  d8 ees f4
  \repeat volta 2 {
    ees8 d c4 
    c bes a c 
    bes8 a g4 f c' 
    c bes a 
  }
  \alternative {
    {
      bes bes8 c d4
    }
    {
      \partial 4 
      d8 c 
    }
  }
  bes c d ees f4 f 
  bes,2 a4 b 
  c4. b8 c4 d 
  ees2. a,4 
  bes ees, f8 f' ees4~
  ees8 d8 c bes f'4 bes, 
  bes4. c8 a4 bes 
  bes a f 
}


baixo = \relative c' {
  \partial 4 
  bes8 a 
  g4 d 
  \repeat volta 2 {
    ees e 
    f c f, a 
    bes c d e 
    f c f, 
  } 
  \alternative {
    {
      d' ees bes
    }
    {
      \partial 4 
      d
    }
  }
  g bes ees, d 
  ees bes f' d 
  c f8 g aes4 g 
  c,2. ees4 
  d c bes c 
  d ees f g 
  f ees2 d8 ees 
  f4 f, bes 
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}