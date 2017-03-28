\version "2.10.33"

\header {
  title = "6 - Christus, der ist mein Leben"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c' {
  \partial 4 f4 
    a4 g4 a4 bes4 
  c2 a4 d4 
  c4 bes4 a4 g4 
  a2 r4 c4 
  d4 e4 f4 e4   %5
  d2 c4 a4 
  bes4 a4 g4 g4 
  f2. 
}

alto = \relative c' {
  \partial 4 c4 
    f4 g4 f4 f4 
  e8 f8 g8 e8 f4 f4 
  f4. e8 f4. e8 
  f2 r4 a4~ 
  a8 g4 c16 bes16 a4 a4   %5
  a4 g4 g4 f4~ 
  f8 e8 f4 f4 e4 
  c2. 
}

tenor = \relative c' {
  \partial 4 a4 
    c4 c4 c4 d4 
  g,4 c4 c4 bes4 
  c4 d8 bes8 c4 c4 
  c2 r4 f4 
  f4 e4. d8 c4~   %5
  c4 b4 e4 c4 
  bes4 c4 des4 c8. bes16 
  a2. 
}

baixo = \relative c, {
  \partial 4 f4 
    f'4 e4 ees4 d4 
  c8 d8 e8 c8 f4 bes,4 
  a4 g4 f8 a8 c4 
  f,2 r4 a'4 
  b,4 c4 d8 f8 a8 g8   %5
  f8 d8 g4 c,4 f,4 
  g4 a4 bes4 c4 
  f,2. 
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