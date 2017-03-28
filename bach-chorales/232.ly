\version "2.10.33"

\header {
  title = "232 - Die Sonn' hat sich mit ihrem Glanz"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \partial 4 d4 
    f4 g4 a4 d4 
  c4. b8 a8 b8 c4 
  b2 a4 d4 
  d4 d4 cis4 e4 
  d4 a4 bes4 a8 g8   %5
  g2 f4 f4 
  e4 e4 d4 a'4 
  c4 c4 g8 a8 bes4 
  a4 a4 c4 d4 
  a4 c4 a4 g8 f8   %10
  e4. d8 d2 
  
}

alto = \relative c' {
  \partial 4 a4 
    d4. e8 f4 a4 
  a4 gis4 a8 gis8 a4 
  a8 gis16 fis16 gis4 e4 a4 
  a8 g8 b4 a4 a8 g8 
  f4 f4 g4 f4   %5
  f8 e16 d16 e4 c4 d4 
  d4 cis4 a4 f'4~ 
  f8 e8 f4 e4 f8 g8 
  f4 f4 f4 f4 
  f4 g4. f16 e16 d4   %10
  d4 cis4 a2 
  
}

tenor = \relative c {
  \partial 4 f4 
    a4 bes4 d4 f4 
  e4 e8. d16 c8 d8 e8 f8 
  b,8 e4 d8 cis4 d4 
  d8 e8 f4 e4 a,4 
  a4 d4. c8 c4   %5
  d8 bes8 g8 c16 bes16 a4 bes4 
  bes4 a8 g8 f4 d'4 
  c4 c4 c8 a8 d8 c8 
  c4 d4 c4. bes8 
  c4 c4 c4 bes4~   %10
  bes8 g8 e8 a16 g16 fis2 
  
}

baixo = \relative c {
  \partial 4 d4 
    d4 g,4 d4 d'4 
  a'4 e4 f4 c8 d8 
  e4 e,4 a4 f'4 
  bes4 a8 gis8 a4 cis,4 
  d8 e8 f4. e8 f4   %5
  bes,4 c4 f,4 bes4 
  g4 a4 d,4 d'4 
  a8 g8 a8 bes8 c4 d8 e8 
  f4 d4 a4 bes4 
  f4 e4 f4 bes4   %10
  g4 a4 d,2 
  
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