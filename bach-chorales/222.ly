\version "2.10.33"

\header {
  title = "222 - Nun preiset alle"
  composer = "J. S. Bach"
}

global =  {
  \time 3/4 
  \key g \major
}

soprano = \relative c {
  g''4 g a 
  b g2 
  a4 a b 
  c2 b4 
  c2.     %% 5
  b4 b cis 
  d a2 
  b4 cis d 
  d2 cis4 
  d2.    %% 10
  a4 a b 
  c d e 
  d2 c4 
  b2 a4
  \repeat volta 2 {
    e'4. d8 c4    %% 15
    d4. c8 b4 
    c c b 
    a2 g4
  }
}


alto = \relative c {
  d'4 e fis 
  g8 f e2 
  f f4 
  g2 g4 
  g2.   %% 5
  g4 g g 
  fis fis2 
  g4 g fis8 g 
  a4 g8 fis g e 
  fis2.   %% 10
  fis4 fis gis 
  a4. gis8 a g 
  fis4 gis a 
  a gis e 
  \repeat volta 2 {
    e8 fis g4 a    %% 15 
    a8 g fis4 g 
    g a g 
    g fis d 
  }
}


tenor = \relative c {
  b'4 b d 
  d c2 
  c d4 
  e2 d8 f 
  e2.    %% 5
  d4 d e 
  a, d2 
  d4 a d4~
  d8 c b4 a 
  a2.    %% 10
  d4 d d 
  e d c8 b 
  a4 b c8 e 
  f4 e8 d c4 
  \repeat volta 2 {
    g' c,8 d e4    %% 115
    d2 d4 
    e d d 
    e d8 c b4 
  }
}


baixo = \relative c {
  g'8 fis e4 d 
  g, c2 
  f4. e8 d4 
  c e g 
  c,2.    %% 5
  g'4 g8 fis e4 
  d2 d4 
  g a b 
  fis g a 
  d,2.    %% 10
  d4 c b 
  a b c4~ 
  c b a 
  d e a, 
  \repeat volta 2 {
    c'4. b8 a g   %% 15
    fis e d4 g8 fis 
    e4 fis g 
    c, d g, 
  }
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