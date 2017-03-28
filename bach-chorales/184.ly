\version "2.10.33"

\header {
  title = "184 - Christ lag in Todesbanden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4  a 
    gis a8 b c4 d 
    c b a a 
    f g a g8 f 
    e d e4 d 
  }  r 
  d8 e f4 g d8 e 
  f4 g a a 
  d cis d e8 d 
  c4 b a b 
  c a8 b c4 g 
  f e d2 
  a'4 g f2 
  e d 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 f4 
    e e e e 
    e e8 d cis4 e 
    d c c8 cis d4 
    d cis a 
  }  r 
  a d d4. cis8 
  d4 c c f8 g 
  a4 g a e 
  e e8 d c4 d 
  e c f e 
  e8 d d cis d2 
  c8 d e4~ e8 cis d4~  
  d cis a2 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d8 c 
    b4 c8 gis a4 b 
    a gis e a 
    a8 bes a g f g a4 
    bes e,8 g f4 
  }  r 
  f8 g a4 g8 a bes4 
  a g f c'
  f e d8 c b4 
  a gis e f 
  g f f c'8 bes 
  a4 g f2 
  f4 e8 a a4 bes~  
  bes a8 g fis2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4  d4 
    e8 d c b a4 gis 
    a e a cis 
    d e f8 e d4 
    g,8 a16 bes a4 d, 
  }  r 
  d'4. c8 bes a g4 
  d' e f f8 e 
  d4 e fis gis 
  a e a, d 
  c f a,8 bes c4 
  d a bes2 
  f'4 cis d bes 
  g a d,2 
  
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