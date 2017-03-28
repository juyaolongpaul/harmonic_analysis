\version "2.10.33"

\header {
  title = "180 - Als Jesus Christus in der Nacht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \partial 4 a4 
    c4 a4 bes4 a4 
  g4 g4 f4 a4 
  b4 c4 d4 c4 
  b2 a4 e4 
  f4 f4 g4 g4   %5
  a4 a4 f4 d4 
  e4 f4 g4 f4 
  e2 d4
  
}

alto = \relative c' {
  \partial 4 f4 
    g4 g8 f8 f8 e8 f4 
  f4 f8 e8 c4 a'4 
  a8 gis8 a4 a8 gis8 a4 
  a4 gis4 e4 cis4 
  d4 d4 d8 b8 cis8 d8   %5
  e8 d8 e8 cis8 d4 a8 b8 
  c4 c4 cis4 d4 
  d4 cis4 a
  
}

tenor = \relative c' {
  \partial 4 d4 
    c4 c4 bes4 c4 
  d4 c8. bes16 a4 c4 
  f4 e4 d4 e4 
  f4 e8 d8 cis4 a4 
  a4 a4 g4 bes4   %5
  a4 a4 a4 f4 
  g4 a8 f8 bes8 g8 a4 
  a8 e8 a8 g8 fis4
  
}

baixo = \relative c {
  \partial 4 d4 
    e4 f4 g4 a4 
  bes4 c4 f,4 f8 e8 
  d4 c4 b4 a4 
  d4 e4 a,4 a4 
  d,8 d'4 c8 b8 e4 d8   %5
  cis8 b8 cis8 a8 d4 d4 
  c8 bes8 a4 e4 f8 g8 
  a2 d4
  
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